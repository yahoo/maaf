# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

from .config import CfgNode

_C = CfgNode()

_C.EXP_NAME = "debug"
_C.OUTPUT_DIR = "./output"

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.COMPOSITION = "maaf"
_C.MODEL.WEIGHTS = None  # load from this path
_C.MODEL.DROPOUT_RATE = 0.1
_C.MODEL.EMBED_DIM = 512
_C.MODEL.LOSS = "batch_based_classification"
_C.MODEL.INCLUDES_IMAGE_TRANSFORM = False
_C.MODEL.INITIAL_NORMALIZATION_FACTOR = 4.0

_C.MODEL.TEXT_MODEL = CfgNode()
_C.MODEL.TEXT_MODEL.ARCHITECTURE = "lstm"
_C.MODEL.TEXT_MODEL.TOKENIZER = "simple"
_C.MODEL.TEXT_MODEL.VOCAB_DATA = None  # path to text file to create vocab
_C.MODEL.TEXT_MODEL.VOCAB_MIN_FREQ = 0
_C.MODEL.TEXT_MODEL.MAX_VOCAB = 52000
_C.MODEL.TEXT_MODEL.TOKENIZER_PATH = None
_C.MODEL.TEXT_MODEL.NUM_LAYERS = 1
_C.MODEL.TEXT_MODEL.FREEZE_WEIGHTS = False
_C.MODEL.TEXT_MODEL.EMBED_DIM = 512  # may need to equal MODEL.EMBED_DIM
_C.MODEL.TEXT_MODEL.OUTPUT_RELU = False
_C.MODEL.TEXT_MODEL.MAX_TOKENS = 128
_C.MODEL.TEXT_MODEL.MODEL_PATH = None

_C.MODEL.IMAGE_MODEL = CfgNode()
_C.MODEL.IMAGE_MODEL.ARCHITECTURE = "resnet50"
_C.MODEL.IMAGE_MODEL.WEIGHTS = None
_C.MODEL.IMAGE_MODEL.PRETRAINED = True
_C.MODEL.IMAGE_MODEL.FREEZE_WEIGHTS = False
_C.MODEL.IMAGE_MODEL.OUTPUTS = [3, 4]

_C.MODEL.MAAF = CfgNode()
_C.MODEL.MAAF.NUM_BLOCKS = 1
_C.MODEL.MAAF.BLOCK_WIDTH = 256
_C.MODEL.MAAF.ATTENTION_HEADS = 8
_C.MODEL.MAAF.POSITION_ENCODING = None
_C.MODEL.MAAF.OUTPUT = "simple_pool"  # rwpool, token
_C.MODEL.MAAF.ATTN_SOFTMAX_REPLACEMENT = None
_C.MODEL.MAAF.RESIDUAL = CfgNode()
_C.MODEL.MAAF.RESIDUAL.LEARN_WEIGHTS = False
_C.MODEL.MAAF.RESIDUAL.INITIAL_MAAF_WEIGHT = 1.
_C.MODEL.MAAF.RESIDUAL.INITIAL_MAAF_PRESIGMOID = None  # deprecated

_C.MODEL.CLIP = CfgNode()
_C.MODEL.CLIP.PROMPT = ""
_C.MODEL.CLIP.MISALIGNMENT = None

# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
_C.DATASET = CfgNode()
_C.DATASET.NAME = "fashioniq"
_C.DATASET.PATH = '/home/default/Data/fashioniq'
_C.DATASET.IMAGE_DIR = ""
_C.DATASET.REQUIRE_IMAGES = False
_C.DATASET.NUM_CLASSES = 3
_C.DATASET.CLASS_WEIGHTS = [1, 1, 1]

_C.DATASET.SINGLE_CLASS_BATCHES = False

_C.DATASET.DPA_ATTRIBUTES = CfgNode()
_C.DATASET.DPA_ATTRIBUTES.DELETE_TERMS = None
_C.DATASET.DPA_ATTRIBUTES.USE_CATEGORY = False

_C.DATASET.CROSS_MODAL = CfgNode()
_C.DATASET.CROSS_MODAL.SOURCE = "title"

_C.DATASET.AUGMENTATION = CfgNode()
_C.DATASET.AUGMENTATION.IMAGE_AUGMENTATION = None

_C.DATA_LOADER = CfgNode()
_C.DATA_LOADER.LOADER_NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()
_C.SOLVER.OPTIMIZER = "sgd"
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.WEIGHT_DECAY = 1e-6
_C.SOLVER.NUM_ITERS = 150000
_C.SOLVER.DROP_WORST_RATE = 0  # 0.2
_C.SOLVER.SOFTMAX_MARGIN = 0

_C.SOLVER.LEARNING_RATE = 1e-2
_C.SOLVER.LEARNING_RATE_DECAY = 0.1
_C.SOLVER.LEARNING_RATE_DECAY_FREQUENCY = 9999999
_C.SOLVER.LR_DECAY_ONLY_ONCE = False
_C.SOLVER.SCHEDULE_RATES = []
_C.SOLVER.SCHEDULE_ITERS = []
_C.SOLVER.PRETRAINED_WEIGHT_LR_FACTOR_TEXT = 1.
_C.SOLVER.PRETRAINED_WEIGHT_LR_FACTOR_IMAGE = 0.1
# By default, projections from image/text model get learning rates including
# the 2 factors above. Setting the parameter below to False changes this.
_C.SOLVER.PROJECTION_LR_TIED_TO_PRETRAINED = True
_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.SAVE_EVERY = 100
_C.SOLVER.EVAL_EVERY = 3
_C.SOLVER.FINAL_EVAL_ON_TEST = False
_C.SOLVER.ALWAYS_EVAL_TEST = False

_C.SOLVER.BATCH_NORM_MODE = "ordinary"
