# 导入必要的库
import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter  # 用于计算平均损失
from torch.cuda import amp  # 自动混合精度训练
import torch.distributed as dist  # 分布式训练
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss  # 监督对比损失函数

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    """
    CLIP-ReID模型第一阶段训练函数
    
    Args:
        cfg: 配置文件对象，包含训练参数
        model: 待训练的模型
        train_loader_stage1: 第一阶段训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        local_rank: 本地GPU编号
    """
    # 从配置文件中获取训练参数
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD  # 检查点保存周期
    device = "cuda"  # 使用GPU设备
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS  # 最大训练轮数
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD  # 日志输出周期 

    # 初始化日志记录器
    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    
    # 将模型移动到GPU设备上
    if device:
        model.to(local_rank)
        # 如果有多个GPU，使用数据并行训练
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    # 初始化训练所需的工具
    loss_meter = AverageMeter()  # 损失值平均计算器
    scaler = amp.GradScaler()    # 混合精度训练的梯度缩放器  
    xent = SupConLoss(device)    # 监督对比损失函数
    
    # 开始训练
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()  # 记录总训练开始时间
    logger.info("model: {}".format(model))
    
    # 第一步：提取所有训练图像的特征向量
    image_features = []  # 存储图像特征
    labels = []          # 存储对应的标签
    
    # 使用no_grad模式提取特征，不计算梯度以节省内存
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)      # 图像数据移到GPU
            target = vid.to(device)   # 标签数据移到GPU
            
            # 使用自动混合精度加速推理
            with amp.autocast(enabled=True):
                # 通过模型获取图像特征向量
                image_feature = model(img, target, get_image = True)  # 获取图像特征
                # 将每个图像特征和对应标签添加到列表中
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())  # 移到CPU以节省GPU内存
                    
        # 将列表转换为张量并移到GPU
        labels_list = torch.stack(labels, dim=0).cuda() #N个标签
        image_features_list = torch.stack(image_features, dim=0).cuda()  # N个图像特征

        # 计算批次相关参数
        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH  # 每批次图像数量
        num_image = labels_list.shape[0]         # 总图像数量
        i_ter = num_image // batch               # 每轮迭代次数
    
    # 删除临时变量释放内存
    del labels, image_features

    # 第二步：正式开始训练循环
    for epoch in range(1, epochs + 1):
        loss_meter.reset()        # 重置损失计算器
        scheduler.step(epoch)     # 更新学习率
        model.train()             # 设置模型为训练模式

        # 随机打乱图像索引，确保每轮训练的随机性
        iter_list = torch.randperm(num_image).to(device)
        
        # 按批次进行训练
        for i in range(i_ter+1):
            optimizer.zero_grad()  # 清零梯度
            
            # 获取当前批次的图像索引
            if i != i_ter:
                # 普通批次：完整的batch_size
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                # 最后一个批次：可能不足batch_size
                b_list = iter_list[i*batch:num_image]
            
            # 根据索引获取当前批次的标签和图像特征
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            
            # 使用自动混合精度计算文本特征
            with amp.autocast(enabled=True):
                text_features = model(label = target, get_text = True)
            
            # 计算双向对比损失
            loss_i2t = xent(image_features, text_features, target, target)  # 图像到文本
            loss_t2i = xent(text_features, image_features, target, target)  # 文本到图像

            loss = loss_i2t + loss_t2i  # 总损失

            # 反向传播和参数更新
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(optimizer)         # 更新参数
            scaler.update()                # 更新缩放因子

            # 更新损失统计
            loss_meter.update(loss.item(), img.shape[0])

            # 同步GPU操作
            torch.cuda.synchronize()
            
            # 定期输出训练日志
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        # 定期保存模型检查点
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                # 分布式训练模式：只有主进程保存模型
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                # 单机训练模式：直接保存模型
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    # 计算并记录总训练时间
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
