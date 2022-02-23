# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


import os
import torch
import datetime
import json
from .datasets.datasets import load_dataset
from tensorboardX import SummaryWriter
import git  # pip install gitpython
from .config.arguments import parse_opt
from .models.build import build_model, get_optimizer
from .config import get_config
from .train import Trainer, MetricTrainer

# avoids a crash on some systems
torch.set_num_threads(1)


def setup(args, modify_exp_name=False):
    cfg = get_config()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.MODEL.DEVICE != "cpu" and not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
    if modify_exp_name:
        curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.EXP_NAME = cfg.EXP_NAME + f"-{curr_time}"
    cfg.freeze()
    return cfg


def main(old_args=False):
    if old_args:
        from .config.compat import compat_setup
        cfg, args = compat_setup()
    else:
        args = parse_opt()
        append_datetime = (args.resume is None) and args.timestamp
        cfg = setup(args, modify_exp_name=append_datetime)

    if args.resume is not None:
        logger = SummaryWriter(args.resume)
    else:
        logger = SummaryWriter(logdir=os.path.join(cfg.OUTPUT_DIR, cfg.EXP_NAME))
        print('Log files saved to', logger.file_writer.get_logdir())

    if args.save_config:
        with open(os.path.join(logger.file_writer.get_logdir(), "config.yaml"),
                  "w") as fh:
            fh.write(cfg.dump())

    # get and save the version of the code being run
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.add_text("git_sha", sha)

    dataset_dict = load_dataset(cfg)
    if cfg.MODEL.TEXT_MODEL.TOKENIZER == "simple":
        texts = dataset_dict["train"].get_all_texts()
    else:
        texts = None
    model, task = build_model(cfg, texts, strict_loading=args.strict_loading)
    optimizer = get_optimizer(cfg, model)

    if args.resume is not None:
        print("loading from: %s" % args.resume)
        loaded_dict = torch.load(
            logger.file_writer.get_logdir() + "/latest_checkpoint.pth")
        model.load_state_dict(loaded_dict["model_state_dict"])
        iteration = loaded_dict["it"]
    else:
        iteration = 0

    if task == "metric":
        trainer = MetricTrainer(cfg, logger, dataset_dict, model, optimizer,
                                iteration)
    else:
        trainer = Trainer(cfg, logger, dataset_dict, model, optimizer,
                          iteration)

    if args.debug:
        import IPython
        IPython.embed()

    if args.train:
        iteration = trainer.train()

    if args.eval:
        results = trainer.run_eval(eval_on_test=args.final_eval_on_test)
        results = {key: val for key, val in results}

        results_file = os.path.join(
            logger.file_writer.get_logdir(),
            f"{cfg.DATASET.NAME}-{iteration}-eval.json")
        with open(results_file, "w") as fh:
            json.dump(results, fh)
        print(f"Evaluation results saved to {results_file}")

        # if cfg.DATASET.NAME == "fashioniq":
        #     print('Generating FashionIQ submission...')
        #     eval_retrieval.predict(cfg, model, dataset_dict["test"],
        #                            filter_categories=True)
        #     print('done')

    logger.close()
