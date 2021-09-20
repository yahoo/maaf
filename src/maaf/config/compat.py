# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


"""Backwards compatibility"""
from .config import CfgNode
from .arguments import old_parse_opt


MAAF_ALIASES = ["sequence_concat_attention", "seqcat_outtoken",
                "concat_attention", "maaf"]

def compat_setup():
    args = old_parse_opt()
    cfg = config_from_args(args).clone()
    cfg.freeze()
    args.resume = args.load
    return cfg, args


def config_from_args(args):
    _C = CfgNode()
    _C.EXP_NAME = args.exp_name
    _C.OUTPUT_DIR = args.savedir

    # ----------------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------------- #
    _C.MODEL = CfgNode()
    _C.MODEL.DEVICE = args.device
    _C.MODEL.COMPOSITION = args.model
    _C.MODEL.WEIGHTS = args.load
    _C.MODEL.DROPOUT_RATE = args.dropout_rate
    _C.MODEL.EMBED_DIM = args.embed_dim
    _C.MODEL.LOSS = args.loss

    _C.MODEL.TEXT_MODEL = CfgNode()
    _C.MODEL.TEXT_MODEL.ARCHITECTURE = \
        None if args.image_only else args.text_model_arch
    _C.MODEL.TEXT_MODEL.TOKENIZER = args.text_tokenizer
    _C.MODEL.TEXT_MODEL.VOCAB_DATA = args.text_data
    _C.MODEL.TEXT_MODEL.VOCAB_MIN_FREQ = args.threshold_rare_words
    _C.MODEL.TEXT_MODEL.MAX_VOCAB = args.max_vocab
    _C.MODEL.TEXT_MODEL.TOKENIZER_PATH = args.tokenizer_path
    _C.MODEL.TEXT_MODEL.NUM_LAYERS = args.text_model_layers
    _C.MODEL.TEXT_MODEL.FREEZE_WEIGHTS = args.freeze_text_model
    _C.MODEL.TEXT_MODEL.EMBED_DIM = args.embed_dim
    _C.MODEL.TEXT_MODEL.OUTPUT_RELU = False

    _C.MODEL.IMAGE_MODEL = CfgNode()
    _C.MODEL.IMAGE_MODEL.ARCHITECTURE = \
        None if args.text_only else args.image_model_arch
    _C.MODEL.IMAGE_MODEL.WEIGHTS = args.image_model_path
    _C.MODEL.IMAGE_MODEL.PRETRAINED = not args.not_pretrained
    _C.MODEL.IMAGE_MODEL.FREEZE_WEIGHTS = args.freeze_img_model
    _C.MODEL.IMAGE_MODEL.OUTPUTS = \
        [ii for ii in range(5) if str(ii) in args.att_layer_spec]

    _C.MODEL.MAAF = CfgNode()
    _C.MODEL.MAAF.NUM_BLOCKS = args.number_attention_blocks
    _C.MODEL.MAAF.BLOCK_WIDTH = args.width_per_attention_block
    _C.MODEL.MAAF.ATTENTION_HEADS = args.number_attention_heads
    _C.MODEL.MAAF.POSITION_ENCODING = args.attn_positional_encoding
    maaf_out = "simple_pool"
    if args.resolutionwise_pool:
        maaf_out = "rwpool"
    if args.model == "seqcat_outtoken":
        maaf_out = "token"
    _C.MODEL.MAAF.OUTPUT = maaf_out
    _C.MODEL.MAAF.ATTN_SOFTMAX_REPLACEMENT = args.attn_softmax_replacement

    # ----------------------------------------------------------------------- #
    # Dataset
    # ----------------------------------------------------------------------- #
    _C.DATASET = CfgNode()
    _C.DATASET.NAME = args.dataset
    _C.DATASET.PATH = args.dataset_path
    _C.DATASET.IMAGE_DIR = args.image_dir
    _C.DATASET.REQUIRE_IMAGES = args.require_images
    _C.DATASET.NUM_CLASSES = args.num_classes
    _C.DATASET.CLASS_WEIGHTS = args.class_weights

    _C.DATASET.SINGLE_CLASS_BATCHES = args.dataset == "fashioniq"

    _C.DATASET.DPA_ATTRIBUTES = CfgNode()
    _C.DATASET.DPA_ATTRIBUTES.DELETE_TERMS = None
    _C.DATASET.DPA_ATTRIBUTES.USE_CATEGORY = False

    _C.DATA_LOADER = CfgNode()
    _C.DATA_LOADER.LOADER_NUM_WORKERS = args.loader_num_workers

    # ----------------------------------------------------------------------- #
    # Solver
    # ----------------------------------------------------------------------- #
    _C.SOLVER = CfgNode()
    _C.SOLVER.BATCH_SIZE = args.batch_size
    _C.SOLVER.WEIGHT_DECAY = args.weight_decay
    _C.SOLVER.NUM_ITERS = args.num_iters
    _C.SOLVER.DROP_WORST_RATE = \
        args.drop_worst_rate if args.drop_worst_flag else 0
    _C.SOLVER.SOFTMAX_MARGIN = args.softmax_margin

    _C.SOLVER.LEARNING_RATE = args.learning_rate
    _C.SOLVER.LEARNING_RATE_DECAY = args.learning_rate_decay
    _C.SOLVER.LEARNING_RATE_DECAY_FREQUENCY = args.learning_rate_decay_frequency
    _C.SOLVER.LR_DECAY_ONLY_ONCE = args.lr_decay_only_once
    _C.SOLVER.SCHEDULE_RATES = args.scheduled_lr_rates
    _C.SOLVER.SCHEDULE_ITERS = args.scheduled_lr_iters
    _C.SOLVER.PRETRAINED_WEIGHT_LR_FACTOR_TEXT = args.pretrained_weight_lr_factor_text
    _C.SOLVER.PRETRAINED_WEIGHT_LR_FACTOR_IMAGE = args.pretrained_weight_lr_factor_image

    _C.SOLVER.SAVE_EVERY = args.save_every
    _C.SOLVER.EVAL_EVERY = args.eval_every

    return _C
