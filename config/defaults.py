from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''

# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'

_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.I2T_LOSS_WEIGHT = 1.0

_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# SIE Parameter
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.TYPE = "clip_vit_base_patch32"
_C.MODEL.BACKBONE.WITH_NORM = True
_C.MODEL.META_ARCHITECTURE = "clip_reid_rga"
_C.MODEL.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
_C.MODEL.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
_C.MODEL.FP16_ENABLED = True
_C.MODEL.RGA = CN()
_C.MODEL.RGA.REGION_NAMES = ["head", "upper body", "lower body", "foot"]
_C.MODEL.RGA.GAMMA = 5.0
_C.MODEL.RGA.DIS_THRESHOLD = 0.85
_C.MODEL.RGA.MEMORY_MOMENTUM = 0.2
_C.MODEL.RGA.NUM_REGIONS = 4
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [224, 224]
# Size of the image during test
_C.INPUT.SIZE_TEST = [224, 224]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 4

# ---------------------------------------------------------------------------- #
# Solver
_C.SOLVER = CN()
# _C.SOLVER.OPT= "Adam"
# _C.SOLVER.MAX_EPOCHS= 120
# _C.SOLVER.BASE_LR= 0.00035
# _C.SOLVER.WEIGHT_DECAY= 5e-4
# _C.SOLVER.BIAS_LR_FACTOR= 2
# _C.SOLVER.IMS_PER_BATCH= 64
# _C.SOLVER.WARMUP_ITERS= 10
# _C.SOLVER.STEPS= [40, 70]
# _C.SOLVER.GAMMA= 0.1
# _C.SOLVER.CHECKPOINT_PERIOD= 10
_C.SOLVER.SEED = 42
# _C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.IMS_PER_BATCH = 8
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.BASE_LR = 0.000005
_C.SOLVER.WARMUP_METHOD = 'linear'
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.MAX_EPOCHS = 60
_C.SOLVER.CHECKPOINT_PERIOD = 60
_C.SOLVER.LOG_PERIOD = 50
_C.SOLVER.EVAL_PERIOD = 60
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.STEPS = [30, 50]
_C.SOLVER.GAMMA = 0.1

# ---------------------------------------------------------------------------- #
# Solver
# _C.SOLVER = CN()
# _C.SOLVER.SEED = 1234
_C.SOLVER.MARGIN = 0.3

# stage1
# ---------------------------------------------------------------------------- #
# Name of optimizer
_C.SOLVER.STAGE1 = CN()

_C.SOLVER.STAGE1.IMS_PER_BATCH = 64

_C.SOLVER.STAGE1.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.STAGE1.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.STAGE1.BASE_LR = 3e-4
# Momentum
_C.SOLVER.STAGE1.MOMENTUM = 0.9

# Settings of weight decay
_C.SOLVER.STAGE1.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE1.WEIGHT_DECAY_BIAS = 0.0005

# warm up factor
_C.SOLVER.STAGE1.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE1.LR_MIN = 0.000016

_C.SOLVER.STAGE1.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE1.WARMUP_METHOD = "linear"

_C.SOLVER.STAGE1.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE1.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.STAGE1.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.STAGE1.LOG_PERIOD = 100
# epoch number of validation
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
# _C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1.EVAL_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Solver
# stage1
# ---------------------------------------------------------------------------- #
_C.SOLVER.STAGE2 = CN()

_C.SOLVER.STAGE2.IMS_PER_BATCH = 64
# Name of optimizer
_C.SOLVER.STAGE2.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.STAGE2.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.STAGE2.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.STAGE2.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.STAGE2.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.STAGE2.MOMENTUM = 0.9
# Margin of triplet loss
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.STAGE2.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.STAGE2.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.STAGE2.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE2.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.STAGE2.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STAGE2.STEPS = (40, 70)
# warm up factor
_C.SOLVER.STAGE2.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.STAGE2.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE2.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE2.LR_MIN = 0.000016


_C.SOLVER.STAGE2.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.STAGE2.WARMUP_METHOD = "linear"

_C.SOLVER.STAGE2.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE2.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.STAGE2.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.STAGE2.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.STAGE2.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch



# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
#_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False

_C.TEST.FEAT_NORM = True
_C.TEST.EVAL_PERIOD = 10
_C.TEST.IMS_PER_BATCH = 64
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""


