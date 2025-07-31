import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import clip
clip_model, _ = clip.load("ViT-B/32", device="cuda")
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts): 
        x = prompts + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) 

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection 
        return x

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 512
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        elif self.model_name =='RGANet':
            self.in_planes = 768
            self.in_planes_proj = 512
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE   

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        self.region_classifier = self.classifier
        self.global_classifier = self.classifier_proj

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual

        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)
        self.img_proj = nn.Linear(768, 512)  # 注意 self.in_planes = 768 for ViT-B-16


    def forward(self, x = None, label=None, get_image = False, get_text = False, cam_label= None, view_label=None):
        if get_text == True:
            prompts = self.prompt_learner(label) 
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features

        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:,0]
        
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label!=None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            image_features_last, image_features, image_features_proj = self.image_encoder(x, cv_embed) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
    
    def extract_feature_map(self, x, cam_label=None, view_label=None):
        """
        提取图像的视觉特征图 F ∈ R^{C×H×W}
        用于计算与文本原型的相似度（Class Segmentation Mask 阶段）
        """
        if self.model_name == 'RN50':
            image_features_last, image_features, _ = self.image_encoder(x)
            F = image_features  # shape: [B, 2048, H, W]
        elif self.model_name == 'ViT-B-16':
            if cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            _, F, _ = self.image_encoder(x, cv_embed)  # shape: [B, num_tokens, dim]
            B, N, C = F.shape
            H = self.h_resolution
            W = self.w_resolution
            F = F[:, 1:, :].permute(0, 2, 1).contiguous().view(B, C, H, W)  # 去掉CLS，变为 [B, C, H, W]
        else:
            raise NotImplementedError(f"Model {self.model_name} not supported in extract_feature_map")

        return F

    def extract_global_feature(self, x, cam_label=None, view_label=None):
        """
        提取全局特征，用于测试阶段 final distance 计算
        """
        if self.model_name == 'RN50':
            _, _, image_features_proj = self.image_encoder(x)
            return image_features_proj[0]
        elif self.model_name == 'ViT-B-16':
            if cam_label is not None and view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label is not None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None
            _, _, image_features_proj = self.image_encoder(x, cv_embed)
            return image_features_proj[:, 0]
        else:
            raise NotImplementedError(f"Model {self.model_name} not supported in extract_global_feature")



def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."

        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = 4
        
        tokenized_prompts = clip.tokenize(ctx_init).cuda() 
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype) 
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors) 

        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])  
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx: , :])  
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    def forward(self, label):
        cls_ctx = self.cls_ctx[label] 
        b = label.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 
    
class RGANet(nn.Module):
    def __init__(self, clip_model, num_classes, region_names=["head", "upper body", "lower body", "foot"]):
        super().__init__()
        self.clip = clip_model
        self.num_regions = len(region_names)
        
        # # 可学习提示设计
        # self.learnable_prompt = nn.ParameterList([
        #     nn.Parameter(torch.randn(512)) for _ in range(8)  # K=8
        # ])
        # self.region_names = region_names
        
        # # 区域评估模块
        # self.discrimination_fc = nn.Linear(512, 1)  # 判别性指标
        # self.register_buffer('memory_bank', torch.zeros(len(region_names), 512))  # 不变性记忆库
        
    def extract_prototypes(clip_model, region_names=["head", "upper body", "lower body", "foot"], K=4):
        """
        使用可学习上下文提示提取区域类别 + 背景的 prototype。
        返回: Tensor [N+1, 512]
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        embed_dim = 512
        context_vectors = nn.Parameter(torch.randn(K, embed_dim)).to(device)
        background_prototype = nn.Parameter(torch.randn(embed_dim)).to(device)
        # printed_once = [False]  # 用列表做可变闭包变量
        def encode_text_with_prompt(class_name):
            tokenized = clip.tokenize(class_name).to(device)
            with torch.no_grad():
                class_embedding = clip_model.token_embedding(tokenized).squeeze(0)  # [token_len, 512]

            prompt_embedding = torch.cat([context_vectors, class_embedding], dim=0)  # [K + token_len, 512]

            # 截断逻辑
            
            max_len = clip_model.positional_embedding.size(0)
            if prompt_embedding.size(0) > max_len:
                # if not printed_once[0]:
                #     print(f"[Warning] Prompt too long: {prompt_embedding.size(0)} > {max_len}. Truncating.")
                #     printed_once[0] = True
                prompt_embedding = prompt_embedding[:max_len, :]
            pos_embed = clip_model.positional_embedding[:prompt_embedding.size(0), :]

            # 添加 batch 维度 + 类型匹配
            prompt_embedding = prompt_embedding.to(clip_model.dtype)
            pos_embed = pos_embed.to(clip_model.dtype)
            x = prompt_embedding.unsqueeze(0) + pos_embed.unsqueeze(0)  # [1, seq_len, 512]

            x = x.permute(1, 0, 2)  # [seq_len, batch, dim]
            with torch.no_grad():
                x = clip_model.transformer(x)
                x = x.permute(1, 0, 2)
                x = clip_model.ln_final(x).type(clip_model.dtype)
                text_feature = x[:, 0, :] @ clip_model.text_projection  # [1, 512]
            return text_feature.squeeze(0)



        # 提取原型
        region_prototypes = []
        for name in region_names:
            p = encode_text_with_prompt(name)
            p = F.normalize(p, dim=0)
            region_prototypes.append(p)

        bg_proto = F.normalize(background_prototype, dim=0)
        region_prototypes.append(bg_proto)

        return torch.stack(region_prototypes, dim=0)  # [N+1, 512]


    # def compute_class_segmentation_mask(image_feature_map, prototypes, img_proj, gamma=5.0):
    #     """
    #     输入：
    #     image_feature_map: [B, d, H, W]，来自 CLIP 的图像特征图
    #     prototypes: [N+1, d]，来自上一步 extract_prototypes() 的文本原型，含 background
    #     gamma: 温度系数（控制 softmax 平滑程度）
    #     输出：
    #     masks: [B, N+1, H, W]，对每个 pixel softmax 后的语义响应图
    #     region_features: [B, N, d]，每类人体区域的特征（masked average pooled）
    #     Args:
    #         image_feature_map: Tensor [B, d, H, W]
    #         prototypes: Tensor [N+1, d]  textual embeddings + background
    #         gamma: float, temperature coefficient

    #     Returns:
    #         masks: Tensor [B, N+1, H, W]  soft class masks
    #         region_features: Tensor [B, N, d]  masked average pooled region features (excluding background)
    #     """
    #     B, d, H, W = image_feature_map.shape
    #     N_plus_1 = prototypes.size(0)

    #     # 归一化图像特征和原型
    #     img_feats = F.normalize(image_feature_map.view(B, d, -1), dim=1)  # [B, d, HW]
    #     protos = F.normalize(prototypes, dim=1).t()  # [d, N+1]
    #     # 如果 img_feats 是 [B, 768, N]
    #     img_feats = img_feats.permute(0, 2, 1)  # [B, N, d]
    #     img_feats = img_proj(img_feats)        # [B, N, 512]
    #     img_feats = img_feats.permute(0, 2, 1)  # [B, 512, N]
    #     # 相似度计算
    #     sim = torch.einsum('bdn,dk->bkn', img_feats, protos)  # [B, N+1, HW]
    #     sim = sim.view(B, N_plus_1, H, W)  # [B, N+1, H, W]

    #     # Softmax归一化为概率掩码
    #     masks = F.softmax(gamma * sim, dim=1)  # [B, N+1, H, W]

    #     # Masked average pooling 得到每个区域的特征（排除背景）
    #     region_feats = []
    #     for j in range(N_plus_1 - 1):  # 只取前N个，排除background
    #         sj = masks[:, j:j+1, :, :]  # [B, 1, H, W]
    #         fj = image_feature_map * sj  # broadcasting 乘法 [B, d, H, W]
    #         pooled = fj.flatten(2).sum(dim=2) / (sj.flatten(2).sum(dim=2) + 1e-6)  # [B, d]
    #         region_feats.append(pooled)

    #     region_feats = torch.stack(region_feats, dim=1)  # [B, N, d]

    #     return masks, region_feats
    def compute_class_segmentation_mask(image_feature_map, prototypes, img_proj, gamma=5.0):
        """
        输入：
            image_feature_map: [B, d, H, W]，来自 CLIP 的图像特征图
            prototypes: [N+1, d]，来自 extract_prototypes() 的文本原型，含 background
            img_proj: nn.Linear，用于将 d 维图像特征映射到 512 维
            gamma: float，softmax 温度系数（越大越尖锐）
        输出：
            masks: [B, N+1, H, W]，对每个像素 softmax 后的语义响应图
            region_features: [B, N, 512]，每类人体区域的 pooled 特征（排除 background）
        """

        B, d, H, W = image_feature_map.shape
        N_plus_1 = prototypes.size(0)

        # --- 特征图归一化并映射到 512 维 ---
        img_feats = image_feature_map.view(B, d, -1)         # [B, 768, HW]
        img_feats = img_feats.permute(0, 2, 1)               # [B, HW, 768]
        img_feats = img_proj(img_feats)                     # [B, HW, 512]
        img_feats = img_feats.permute(0, 2, 1)               # [B, 512, HW]

        # --- 原型归一化并转置 ---
        protos = F.normalize(prototypes, dim=1).t()                      # [d or 512, N+1]

        # --- 相似度计算并 reshape ---
        sim = torch.einsum('bdn,dk->bkn', img_feats, protos)             # [B, N+1, HW]
        sim = sim.view(B, N_plus_1, H, W)                                # [B, N+1, H, W]

        # --- softmax 得到响应图（mask） ---
        masks = F.softmax(gamma * sim, dim=1)                            # [B, N+1, H, W]

        # --- 映射 image_feature_map 为 512 维以做区域 pooling ---
        proj_feature_map = img_proj(image_feature_map.flatten(2).permute(0, 2, 1))  # [B, HW, 512]
        proj_feature_map = proj_feature_map.permute(0, 2, 1).view(B, 512, H, W)     # [B, 512, H, W]

        # --- masked average pooling 得到每类区域的特征 ---
        region_feats = []
        for j in range(N_plus_1 - 1):  # 只取前 N 个类，不包括 background
            sj = masks[:, j:j+1, :, :]                      # [B, 1, H, W]
            fj = proj_feature_map * sj                      # [B, 512, H, W]
            pooled = fj.flatten(2).sum(dim=2) / (sj.flatten(2).sum(dim=2) + 1e-6)  # [B, 512]
            region_feats.append(pooled)

        region_feats = torch.stack(region_feats, dim=1)     # [B, N, 512]

        return masks, region_feats


    def compute_discrimination_scores(region_features):
        """
        Args:
            region_features: Tensor [B, N, d] -- 每个区域的全局特征表示

        Returns:
            scores: Tensor [B, N] -- 每个区域的判别性得分 α_j ∈ (0, 1)
        """
        B, N, d = region_features.shape
        linear = nn.Linear(d, 1).to(region_features.device)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0)

        scores = linear(region_features)  # [B, N, 1]
        scores = torch.sigmoid(scores).squeeze(-1)  # [B, N]
        return scores


    def initialize_memory_bank(fixed_stripe_feats):
        """
        使用初始条带特征（如horizontal stripe）初始化区域记忆中心
        fixed_stripe_feats: Tensor [N, d] 初始每类区域平均特征
        """
        return F.normalize(fixed_stripe_feats.clone().detach(), dim=1)  # [N, d]


    def update_memory_bank(memory_bank, region_features, discrimination_scores, threshold=0.85, momentum=0.2):
        """
        只使用高α区域更新 memory bank
        region_features: [B, N, d]
        discrimination_scores: [B, N]
        memory_bank: Tensor [N, d]
        """
        B, N, d = region_features.shape
        for j in range(N):
            # mask: 哪些样本中该区域得分 > threshold
            mask = discrimination_scores[:, j] > threshold
            if mask.sum() == 0:
                continue
            selected_feats = region_features[mask, j, :]  # [B_j, d]
            if selected_feats.shape[0] == 0:
                continue
            with torch.no_grad():   # 防止出现梯度问题
                mean_feat = F.normalize(selected_feats.mean(dim=0), dim=0)  # [d]
                memory_bank[j] = momentum * memory_bank[j] + (1 - momentum) * mean_feat
                # memory_bank[j] = F.normalize(memory_bank[j], dim=0)
                
                memory_bank[j] = F.normalize(memory_bank[j].clone(), dim=0)



    def compute_invariance_scores(region_features, memory_bank, img_proj):
        """
        region_features: [B, N, d]
        memory_bank: [N, d]
        return: invariance_scores: [B, N]
        """
        region_features = F.normalize(region_features, dim=2)
        memory_bank = F.normalize(memory_bank, dim=1)  # [N, d]
        sim = torch.einsum('bnd,nd->bn', region_features, memory_bank)  # cosine similarity
        inv_scores = F.softmax(sim, dim=1)  # [B, N]
        return inv_scores


    def combine_confidence_scores(alpha, beta):
        """
        综合判别性 α_j 和不变性 β_j 得到最终区域置信度分数 w_j ∈ [0, 1]
        alpha: [B, N]
        beta:  [B, N]
        return: w: [B, N]
        """
        w = alpha + beta
        w = F.softmax(w, dim=1)
        return w


    def region_weighted_reid_loss(region_features, labels, classifier, weights):
        """
        计算 Region-Aware Loss
        region_features: [B, N, d]
        labels: [B] int64
        classifier: nn.ModuleList or shared nn.Linear(d, num_classes)
        weights: [B, N] confidence score w_j
        return: scalar loss
        """
        B, N, d = region_features.shape
        losses = []
        for j in range(N):
            feats_j = region_features[:, j, :]  # [B, d]
            logits_j = classifier(feats_j)      # [B, num_classes]
            log_probs = F.log_softmax(logits_j, dim=1)  # [B, C]
            ce = F.nll_loss(log_probs, labels, reduction='none')  # [B]
            weighted = ce * weights[:, j]  # [B]
            losses.append(weighted.mean())
        return sum(losses) / N


    def global_cross_entropy_and_triplet(global_features, labels, classifier):
        logits = classifier(global_features)
        ce_loss = F.cross_entropy(logits, labels)
        # Triplet loss placeholder - use a real one like from torchreid or custom
        triplet_loss = torch.tensor(0.0, device=global_features.device)
        return ce_loss + triplet_loss


    def total_training_loss(region_features, global_features, labels, region_classifier, global_classifier, region_weights):
        """
        计算总损失: 区域交叉熵 + 区域triplet + 全局交叉熵 + 全局triplet
        """
        
        L_ram = RGANet.region_weighted_reid_loss(region_features, labels, region_classifier, region_weights)
        # Triplet loss placeholder, replace with actual computation if needed
        L_tri_region = torch.tensor(0.0, device=region_features.device, requires_grad=True)
        for j in range(region_features.size(1)):
            pass  # 可在此循环内加 Triplet(F^r_j) 损失计算

        L_global = RGANet.global_cross_entropy_and_triplet(global_features, labels, global_classifier)
        return L_ram + L_tri_region + L_global


    def compute_region_distance(query_feats, gallery_feats):
        """
        计算每个区域的余弦距离
        query_feats, gallery_feats: [B, N, d]
        return: [B, N]
        """
        query_feats = F.normalize(query_feats, dim=2)
        gallery_feats = F.normalize(gallery_feats, dim=2)
        return 1 - torch.einsum('bnd,bnd->bn', query_feats, gallery_feats)


    def compute_final_matching_distance(query_feats, gallery_feats, query_global, gallery_global, w_query, w_gallery):
        """
        加权融合局部距离和全局距离
        """
        region_d = RGANet.compute_region_distance(query_feats, gallery_feats)  # [B, N]
        global_d = 1 - F.cosine_similarity(F.normalize(query_global, dim=1), F.normalize(gallery_global, dim=1))  # [B]

        weights = w_query * w_gallery  # [B, N]
        w_sum = weights.sum(dim=1) + 1e-6
        final_d = (region_d * weights).sum(dim=1) + global_d  # [B]
        final_d = final_d / (w_sum + 1)
        return final_d