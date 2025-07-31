# RGANet-CLIP
- train: CUDA_VISIBLE_DEVICES=0 python train_rga.py --config_file configs/person/rga_vit.yml
- test: CUDA_VISIBLE_DEVICES=0 python test_rga.py --config_file configs/person/rga_vit.yml TEST.WEIGHT 'your_path/ViT-B-16_60.pth'
