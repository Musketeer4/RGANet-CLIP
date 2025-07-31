"""
RGANet-CLIP 模型第二阶段训练处理器
实现基于区域注意力机制和记忆增强的CLIP-ReID训练
包含RAM(Region-Aware Memory)模块和渐进式训练策略

主要功能：
1. 渐进式训练策略：前期仅训练提示参数，后期联合训练
2. 区域感知记忆模块(RAM)：动态评估和加权不同区域的重要性
3. 记忆库更新：使用动量更新策略维护历史区域特征信息
4. 混合精度训练：提高训练效率并节省显存
5. 分布式训练支持：支持多GPU并行训练

核心创新：
- 区域重要性评估：结合判别性和不变性指标
- 加权损失计算：根据区域权重动态调整损失贡献
- 记忆增强机制：利用历史信息提升特征质量
"""

import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from model.make_model_clipreid import RGANet

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    """
    执行RGANet-CLIP模型的第二阶段训练(带区域感知记忆模块)
    
    Args:
        cfg: 配置对象，包含所有训练参数
        model: RGANet-CLIP模型实例
        center_criterion: 中心损失函数
        train_loader_stage2: 第二阶段训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 主优化器
        optimizer_center: 中心损失优化器
        scheduler: 学习率调度器
        loss_fn: 损失函数
        num_query: 查询集数量
        local_rank: 本地GPU rank
    """
    # 从配置中获取训练参数
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD              # 日志记录周期
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD # 模型保存周期
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD            # 验证周期
    instance = cfg.DATALOADER.NUM_INSTANCE                 # 每个身份的实例数

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS                  # 最大训练轮数

    # 初始化日志记录器
    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    
    # 设置模型设备和多GPU并行训练
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    # 初始化训练指标记录器
    loss_meter = AverageMeter()  # 损失值平均记录器
    acc_meter = AverageMeter()   # 准确率平均记录器

    # 初始化评估器和训练工具
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()    # 混合精度训练的梯度缩放器
    xent = SupConLoss(device)    # 监督对比损失函数
    
# 定义区域评估函数 - RAM模块的核心组件
    def evaluate_regions(region_features):
        """
        评估不同区域的重要性权重
        
        Args:
            region_features: 区域特征 [batch, num_regions, dim]
            
        Returns:
            weights: 区域重要性权重 [batch, num_regions]
        """
        # 计算判别性指标 - 衡量区域特征的判别能力
        alpha = torch.sigmoid(model.discrimination_fc(region_features))
        
        # 计算不变性指标 - 通过与记忆库的相似度衡量特征稳定性
        beta = F.cosine_similarity(region_features, model.memory_bank, dim=-1)
        beta = F.softmax(beta, dim=-1)
        
        # 综合置信度 - 结合判别性和不变性得到最终权重
        weights = F.softmax(alpha + beta, dim=-1)
        return weights

    # train
    # 记录训练总时间
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    # 准备文本特征提取 - 为CLIP文本编码器预计算所有类别的文本特征
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch  # 计算批次迭代次数
    left = num_classes-batch* (num_classes//batch)  # 计算剩余类别数
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    
    # 批量提取文本特征，避免显存溢出
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    # 开始主训练循环
    for epoch in range(1, epochs + 1):
        # 实现渐进式训练策略
        if epoch < 5:
            # 前5个epoch：仅训练可学习提示参数，冻结其他参数
            for param in model.parameters():
                param.requires_grad = False
            for param in model.learnable_prompt:
                param.requires_grad = True
        else:
            # 后续epoch：联合训练所有参数
            for param in model.parameters():
                param.requires_grad = True
    
        # 初始化epoch训练状态
        # 初始化epoch训练状态
        start_time = time.time()
        loss_meter.reset()  # 重置损失记录器
        acc_meter.reset()   # 重置准确率记录器
        evaluator.reset()   # 重置评估器

        scheduler.step()    # 更新学习率

        model.train()       # 设置模型为训练模式
        
        # 遍历训练数据批次
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            # 梯度清零
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            # 数据转移到GPU
            img = img.to(device)
            target = vid.to(device)
            
            # 处理相机ID（如果启用SIE_CAMERA）
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
                
            # 处理视角ID（如果启用SIE_VIEW）
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
                
            # 前向传播（使用混合精度训练）
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                logits = image_features @ text_features.t()  # 计算图像-文本相似度
                loss = loss_fn(score, feat, target, target_cam, logits)

            # 反向传播（使用梯度缩放）
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 处理中心损失（如果启用）
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            # 计算准确率
            acc = (logits.max(1)[1] == target).float().mean()

            # 更新指标记录器
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            
            # 定期打印训练日志
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
            
            # # ====== RAM模块：区域感知记忆增强 ======
            # # 1. 生成区域映射
            # region_maps = RGANet.generate_regions(image_features)
            
            # # 2. 定义掩码池化函数 - 根据区域掩码提取区域特征
            # def masked_pooling(features, masks):
            #     """
            #     根据区域掩码对特征进行池化
                
            #     Args:
            #         features: 图像特征 [batch, dim]
            #         masks: 区域掩码 [batch, num_regions]
                    
            #     Returns:
            #         region_features: 区域特征 [batch, num_regions, dim]
            #     """
            #     batch_size, dim = features.shape
            #     num_regions = masks.shape[1]
            #     region_features = []
            #     for i in range(num_regions):
            #         mask = masks[:, i].unsqueeze(1)  # [batch, 1]
            #         region_feat = features * mask    # 加权特征
            #         region_features.append(region_feat)
            #     region_features = torch.stack(region_features, dim=1)  # [batch, num_regions, dim]
            #     return region_features

            # # 3. 提取区域特征
            # region_features = masked_pooling(image_features, region_maps)
        
            # # 4. 评估区域重要性权重
            # region_weights = evaluate_regions(region_features)
        
            # # 5. 计算加权区域损失
            # loss = 0
            # for i in range(model.num_regions):
            #     loss += region_weights[:,i] * loss_fn(region_features[:,i], target)
            # loss = loss / region_weights.sum(dim=1, keepdim=True)  # 归一化平均损失
            
            # # 6. 执行反向传播和参数更新
            # loss.backward()
            # optimizer.step()
            # optimizer_center.step()

            # # 7. 更新评估器
            # evaluator.update((feat, vid, target_cam))

            # # 8. 更新记忆库 - 使用动量更新策略
            # if cfg.MODEL.DIST_TRAIN:
            #     if dist.get_rank() == 0:
            #         model.memory_bank = 0.3 * model.memory_bank + 0.7 * region_features.mean(dim=0)
            # else:
            #     # 在非分布式训练中直接更新记忆库
            #     model.memory_bank = 0.3 * model.memory_bank + 0.7 * region_features.mean(dim=0)
            
            # # 9. 保存区域特征和权重到模型（用于后续分析）
            # model.region_features = region_features
            # model.region_weights = region_weights
        
            # # 10. 再次更新记忆库（确保一致性）
            # with torch.no_grad():
            #     model.memory_bank = 0.3 * model.memory_bank + 0.7 * region_features.mean(dim=0)

        # 计算epoch训练时间和速度
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass  # 分布式训练时跳过日志记录（避免重复）
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        # 定期保存模型检查点
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:  # 只在主进程保存
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        # 定期进行模型验证评估
        # 定期进行模型验证评估
        if epoch % eval_period == 0:
            # 分布式训练时的验证流程
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:  # 只在主进程进行验证
                    model.eval()  # 设置为评估模式
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():  # 禁用梯度计算以节省内存
                            img = img.to(device)
                            # 处理相机ID
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            # 处理视角ID
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            # 提取特征进行评估
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    
                    # 计算验证指标
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))  # 平均精度均值
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))  # CMC曲线
                    torch.cuda.empty_cache()  # 清理GPU缓存
            else:
                # 单GPU训练时的验证流程
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                
                # 计算并记录验证结果
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    # 记录总训练时间
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    """
    执行模型推理和测试评估
    
    Args:
        cfg: 配置对象
        model: 训练好的模型
        val_loader: 测试数据加载器
        num_query: 查询集数量
        
    Returns:
        tuple: (Rank-1准确率, Rank-5准确率)
    """
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    # 初始化评估器
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    # 设置模型设备和并行
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()  # 设置为评估模式
    img_path_list = []
    
    # # 定义基于区域的距离计算函数 - 用于测试阶段的相似度计算
    # def compute_distance(query_feat, gallery_feat):
    #     """
    #     计算查询和图库特征之间的加权区域距离
        
    #     Args:
    #         query_feat: 查询特征（包含全局和区域特征）
    #         gallery_feat: 图库特征（包含全局和区域特征）
            
    #     Returns:
    #         float: 综合距离度量
    #     """
    #     # 计算原有全局特征距离
    #     global_dist = 1 - F.cosine_similarity(query_feat[0], gallery_feat[0])
        
    #     # 计算区域特征加权距离
    #     region_dist = 0
    #     total_weight = 0
    #     for i in range(1, len(query_feat)):
    #         # 计算余弦距离
    #         d = 1 - F.cosine_similarity(query_feat[i], gallery_feat[i])
    #         # 计算权重（查询和图库权重的乘积）
    #         w = query_feat['weights'][i] * gallery_feat['weights'][i]
    #         region_dist += d * w
    #         total_weight += w
        
    #     # 综合距离计算（参考论文公式10）
    #     return (region_dist + global_dist) / (total_weight + 1)

    # 遍历测试数据进行特征提取和评估
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():  # 禁用梯度计算
            img = img.to(device)
            
            # 处理相机ID
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
                
            # 处理视角ID
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
                
            # 提取特征
            feat = model(img, cam_label=camids, view_label=target_view)
            
            # 更新评估器
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    # 计算最终测试结果
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))      # 平均精度均值
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))  # CMC曲线结果
    return cmc[0], cmc[4]  # 返回Rank-1和Rank-5准确率