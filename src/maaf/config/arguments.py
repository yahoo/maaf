# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--config_file', type=str, default="")
    add_arg('--no-train', action="store_false", dest="train")
    add_arg('--no-eval', action="store_false", dest="eval",
            help="skip the eval following training loop")
    add_arg('--no-config-save', action="store_false", dest="save_config")
    add_arg('--debug', action="store_true")
    add_arg('--resume', type=str, default=None)
    add_arg('--final_eval_on_test', action="store_true")
    add_arg('--non-strict_loading', action="store_false", dest="strict_loading")
    add_arg('--no-timestamp', action="store_false", dest="timestamp")

    add_arg(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. ",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def parse_opt():
    args = get_parser().parse_args()
    return args


def old_parser():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-f', type=str, default='')
    add_arg('--exp_name', type=str, default='debug')
    add_arg('--comment', type=str, default='')
    add_arg('--savedir', type=str, default='')

    add_arg('--device', type=str, default='cuda')

    add_arg('--dataset', type=str, default='dpa_gender')
    add_arg('--dataset_path', type=str,
            default='')
    add_arg('--image_dir', type=str,
            default="")
    add_arg('--require_images', action='store_true',
            help="for datasets where some examples have images and some do not")
    add_arg('--text_only', action='store_true')
    add_arg('--image_only', action='store_true')

    add_arg('--text_tokenizer', type=str, default="simple")
    add_arg('--text_data', type=str, default=None,
            help="if building vocab/tokenizer from data, a file with all text")
    add_arg('--max_vocab', type=int, default=52000)
    add_arg('--tokenizer_path', type=str, default="")

    add_arg('--class_weights', nargs='+', type=float, default=[23, 2.5, 6.4],
            help="default is inverse frequency for unisex, female, male in apparel dataset")

    add_arg('--model', type=str, default='maaf')
    add_arg('--embed_dim', type=int, default=512)
    add_arg('--batch_size', type=int, default=32)
    add_arg('--weight_decay', type=float, default=1e-6)
    add_arg('--num_iters', type=int, default=150000)
    add_arg('--loss', type=str, default='batch_based_classification')
    add_arg('--num_classes', type=int, default=3)

    add_arg('--learning_rate', type=float, default=1e-2)
    add_arg('--learning_rate_decay', type=float, default=0.1)
    add_arg('--learning_rate_decay_frequency', type=int, default=9999999)
    add_arg('--lr_decay_only_once', action="store_true")
    # more flexible learning rate scheduling.
    # both args below must be set or we default to old scheme
    add_arg('--scheduled_lr_rates', type=str, default="",
            help="Separate rates by commas." +
            "The learning_rate argument sets the initial rate; " +
            "this param sets rates after each scheduled_lr_iters entry" +
            "If empty string, old regular decay schedule is used.")
    add_arg('--scheduled_lr_iters', type=str, default="",
        help="Separate iteration numbers by commas." +
             "If empty string, old regular decay schedule is used.")
    add_arg('--pretrained_weight_lr_factor_image', type=float, default=0.1)
    add_arg('--pretrained_weight_lr_factor_text', type=float, default=1.)

    add_arg('--loader_num_workers', type=int, default=4)

    add_arg('-t', "--test_only", action="store_true")
    add_arg('-l', '--load', type=str, default="")

    add_arg('--dropout_rate', type=float, default=0.1)

    add_arg('--drop_worst_flag', action='store_true',
            help='If added the model will ingore the highest --drop_worst_rate losses')
    add_arg('--drop_worst_rate', type=float, default=0.2)

    add_arg('--image_model_arch', type=str, default='resnet50')
    add_arg('--image_model_path', type=str, default='')
    add_arg('--not_pretrained', action='store_true',
            help='If added, the image network will be trained WITHOUT ImageNet-pretrained weights.')
    add_arg('--freeze_img_model', action='store_true',
            help='If added the loaded image model weights will not be finetuned')

    add_arg('--text_model_arch', type=str, default='lstm')
    add_arg('--text_model_layers', type=int, default=1)
    add_arg('--normalize_text', action='store_true')
    add_arg('--threshold_rare_words', type=int, default=0)
    add_arg('--freeze_text_model', action='store_true',
            help='If added the loaded text model weights will not be finetuned')

    add_arg('--number_attention_blocks', type=int, default=1)
    add_arg('--width_per_attention_block', type=int, default=256)
    add_arg('--number_attention_heads', type=int, default=8)
    add_arg('--attn_positional_encoding', default=None)
    add_arg('--resolutionwise_pool', action='store_true')
    add_arg('--attn_softmax_replacement', type=str, default="none")
    add_arg('--att_layer_spec', type=str, default="3_4")

    add_arg('--softmax_margin', type=float, default=0)

    add_arg('--save_every', type=int, default=100,
            help="keep checkpoints this often in epochs")
    add_arg('--eval_every', type=int, default=3,
            help="run eval on val set this often in epochs")
    add_arg('--final_eval_on_test', action="store_true")

    add_arg('--inspect_after', action="store_true")

    return parser


def old_parse_opt():
    args, unknown = old_parser().parse_known_args()
    if args.load == "":
        args.load = None
    if args.image_model_path in ["", "none", "None"]:
        args.image_model_path = None
    if args.image_model_arch in ["", "none", "None"]:
        args.image_model_arch = None
    args.eval_only = args.test_only
    if args.attn_softmax_replacement == "none":
        args.attn_softmax_replacement = None
    args.debug = args.inspect_after

    for unk in unknown:
        print(f"WARNING: unrecognized argument: {unk}")

    return args
