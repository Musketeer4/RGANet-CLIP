"""
处理器模块：包含模型训练和推理的核心逻辑

该模块提供了深度学习模型的训练和推理功能，主要用于人员重识别（Person Re-ID）任务。
支持多GPU训练、混合精度训练、模型评估等功能。
"""

import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter  # 平均值计算工具
from utils.metrics import R1_mAP_eval  # 评估指标计算器（Rank-1准确率和mAP）
from torch.cuda import amp  # 自动混合精度训练
import torch.distributed as dist  # 分布式训练支持

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    """
    执行模型训练的主函数
    
    Args:
        cfg: 配置对象，包含所有训练参数
        model: 要训练的神经网络模型
        center_criterion: 中心损失函数准则
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 主优化器
        optimizer_center: 中心损失优化器
        scheduler: 学习率调度器
        loss_fn: 损失函数
        num_query: 查询样本数量
        local_rank: 本地GPU排名（用于分布式训练）
    """
    # 从配置文件中获取训练相关参数
    log_period = cfg.SOLVER.LOG_PERIOD  # 日志记录周期
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD  # 模型保存周期
    eval_period = cfg.SOLVER.EVAL_PERIOD  # 验证周期

    device = "cuda"  # 指定使用CUDA设备
    epochs = cfg.SOLVER.MAX_EPOCHS  # 最大训练轮数

    # 初始化日志记录器
    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    
    # 模型设备配置和多GPU设置
    if device:
        model.to(local_rank)  # 将模型移动到指定GPU
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  # 使用数据并行进行多GPU训练

    # 初始化训练过程中的度量器
    loss_meter = AverageMeter()  # 损失值平均计算器
    acc_meter = AverageMeter()   # 准确率平均计算器

    # 初始化评估器，用于计算Rank-1准确率和mAP
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()  # 混合精度训练的梯度缩放器
    
    # 开始训练循环
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()  # 记录总训练开始时间
    logger.info("model: {}".format(model))

    # 主训练循环：逐轮次训练
    for epoch in range(1, epochs + 1):
        start_time = time.time()  # 记录当前轮次开始时间
        # 重置度量器以准备新的轮次
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()  # 更新学习率

        model.train()  # 设置模型为训练模式
        # 遍历训练数据批次
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            # 清空梯度
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            # 将数据移动到GPU
            img = img.to(device)
            target = vid.to(device)  # vid是身份标签
            
            # 处理相机标签（如果启用SIE_CAMERA）
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
                
            # 处理视角标签（如果启用SIE_VIEW）
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
                
            # 前向传播（使用混合精度）
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            # 反向传播（使用梯度缩放）
            scaler.scale(loss).backward()

            # 更新主优化器参数
            scaler.step(optimizer)
            scaler.update()

            # 如果使用中心损失，更新中心损失优化器
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
                
            # 计算准确率
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            # 更新损失和准确率度量器
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            # 同步GPU操作
            torch.cuda.synchronize()
            
            # 定期记录训练日志
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        # 计算当前轮次的训练时间和速度
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass  # 分布式训练模式下的处理（当前为空）
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # 定期保存模型检查点
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                # 分布式训练：只有rank 0进程保存模型
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                # 单机训练：直接保存模型
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        # 定期进行模型验证评估
        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                # 分布式训练：只有rank 0进程进行验证
                if dist.get_rank() == 0:
                    model.eval()  # 切换到评估模式
                    # 遍历验证数据集
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():  # 验证时不计算梯度
                            img = img.to(device)
                            # 处理相机标签
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            # 处理视角标签
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            # 前向传播获取特征
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    
                    # 计算并记录验证结果
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()  # 清理GPU缓存
            else:
                # 单机训练的验证过程
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

    # 训练完成，记录总训练时间
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)  # 打印输出目录路径


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    """
    执行模型推理的函数
    
    Args:
        cfg: 配置对象，包含推理相关参数
        model: 已训练好的模型
        val_loader: 验证/测试数据加载器
        num_query: 查询样本数量
        
    Returns:
        tuple: 返回Rank-1和Rank-5准确率
    """
    device = "cuda"  # 指定使用CUDA设备
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    # 初始化评估器
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()  # 重置评估器状态

    # 模型设备配置
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  # 多GPU推理
        model.to(device)  # 将模型移动到GPU

    model.eval()  # 设置模型为评估模式
    img_path_list = []  # 存储图像路径列表（未使用）

    # 遍历测试数据进行推理
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():  # 推理时不计算梯度
            img = img.to(device)
            
            # 处理相机标签
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
                
            # 处理视角标签
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
                
            # 前向传播获取特征
            feat = model(img, cam_label=camids, view_label=target_view)
            # 更新评估器，传入特征、人员ID和相机ID
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)  # 记录图像路径


    # 计算最终的评估指标
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))  # 记录mAP（平均精度均值）
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))  # 记录CMC曲线的Rank-1, 5, 10准确率
    return cmc[0], cmc[4]  # 返回Rank-1和Rank-5准确率