import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.make_model_clipreid import RGANet
from utils.metrics import R1_mAP_eval  # 评估指标计算工具
import torch.distributed as dist

def train_rga(cfg, model, train_loader, val_loader, optimizer, scheduler, num_query, num_classes, clip_model):
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    print("Starting RGA training...")
    torch.autograd.set_detect_anomaly(True) # 开启异常检测，帮助调试梯度问题

    # clip_model = model.clip  # 使用ViT/ResNet视觉编码器
    region_names = ["head", "upper body", "lower body", "foot"]
    
    # 初始化 memory bank
    print("Initializing memory bank with random values...")
    memory_bank = RGANet.initialize_memory_bank(torch.randn(len(region_names), 512).cuda())
    
    # 检测2
    sample = train_loader.dataset[0]
    print("Got one sample:", type(sample), len(sample))

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        print(f"Epoch [{epoch + 1}/{cfg.SOLVER.MAX_EPOCHS}]")
        model.train()
        # 检测1
        # print("Checking data loading...")
        # for i, (images, labels, _, _) in enumerate(train_loader):
        #     print(f"Batch {i}: images.shape = {images.shape}, labels.shape = {labels.shape}")
        #     if i >= 2:
        #         break
        for images, labels, _, _ in train_loader:
            images = images.cuda()
            labels = labels.cuda()
            # print("Moved to CUDA")

            # 提取图像特征图
            # F = model.extract_feature_map(images)  # output: [B, 512, H, W]
            F = model.extract_feature_map(images)
            # print("Extracted feature map")

            # Prototype 提取
            prototypes = RGANet.extract_prototypes(clip_model, region_names)
            # print("Extracted prototypes")

            # 生成分割 mask 和区域特征
            device = F.device  # F 是 image_feature_map，shape: [B, d, H, W]
            model.img_proj = model.img_proj.to(device)
            masks, region_features = RGANet.compute_class_segmentation_mask(F, prototypes, model.img_proj)
            # print("Computed masks and region features")

            # 计算判别性得分 α_j
            alpha = RGANet.compute_discrimination_scores(region_features)
            # print("Computed alpha (discriminability)")

            
            # 更新 memory bank
            RGANet.update_memory_bank(memory_bank, region_features, alpha)
            # print("Updated memory bank")

            # 计算不变性得分 β_j
            beta = RGANet.compute_invariance_scores(region_features, memory_bank, model.img_proj)
            # print("Computed beta (invariance)")

            
            # 得到最终置信度 w_j
            w = RGANet.combine_confidence_scores(alpha, beta)
            # print("Combined confidence scores to get w")

            
            # 提取 global feature
            global_feat = model.extract_global_feature(images)
            # print("Extracted global feature")

            
            # 计算总损失
            loss = RGANet.total_training_loss(region_features, global_feat, labels, model.region_classifier, model.global_classifier, w)
            # print("Computed total training loss")

            
            # 优化器更新
            optimizer.zero_grad()
            # print("Zeroed optimizer gradients")

            loss.backward()
            # print("Backpropagated loss")

            optimizer.step()
            if epoch % checkpoint_period == 0:
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(),
                                os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                else:
                    torch.save(model.state_dict(),
                            os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        
        scheduler.step(epoch)

        # TODO: 验证阶段调用 compute_final_matching_distance 进行匹配并评估
def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    """
    模型推理函数，用于在验证集上评估训练好的模型
    
    Args:
        cfg: 配置对象
        model: 训练好的模型
        val_loader: 验证数据加载器
        num_query: 查询集数量
        
    Returns:
        tuple: 返回Rank-1和Rank-5准确率
    """
    device = "cuda"  # 使用GPU设备
    
    # 初始化日志记录器
    logger = logging.getLogger("RGA")
    logger.info("Enter inferencing")
    
    # 将模型移至GPU设备
    model.to(device)

    # 初始化评估器
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    # 设置模型为评估模式
    model.eval()
    
    print("Starting inference on validation set...")
    # 遍历验证数据集进行推理
    for n_iter, (img, pid, camid, _) in enumerate(val_loader):
        with torch.no_grad():  # 禁用梯度计算
            img = img.to(device)
            
            # 根据配置决定是否使用相机信息
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
                
            # 根据配置决定是否使用视角信息
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
                
            # 提取特征
            feat = model(img, cam_label=camids, view_label=target_view)
            # 更新评估器
            evaluator.update((feat, pid, camid))

    # 计算最终评估指标
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    
    # 输出推理结果
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        
    # 返回Rank-1和Rank-5准确率
    return cmc[0], cmc[4]
