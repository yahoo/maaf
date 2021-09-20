# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


from maaf.main import setup
from maaf.models.build import build_model
from maaf.datasets.cfq import CFQSet
from maaf.datasets.datasets import get_default_image_transform
import argparse
import os
import json
import torch


def parse_opt():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg('--config_file', type=str)
    add_arg('--weights_path', type=str, default=None)
    add_arg('--debug', action="store_true")
    add_arg('--non-strict_loading', action="store_false", dest="strict_loading")
    add_arg('--data_path', default="/home/default/ephemeral_drive/Data/cfq/")
    add_arg('--output_path', type=str, default=None)

    add_arg(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. ",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def evaluate():
    args = parse_opt()
    cfg = setup(args)

    model, task = build_model(cfg, None, strict_loading=args.strict_loading)
    if args.weights_path is None:
        weights_path = os.path.dirname(args.config_file)
        weights_path = os.path.join(weights_path, "latest_checkpoint.pth")
    else:
        weights_path = args.weights_path
    if not os.path.exists(weights_path):
        print(f"Checkpoint {weights_path} not found, evaluating without")
    else:
        print(f"Loading from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=model.device)["model_state_dict"]
        model.load_state_dict(state_dict, strict=args.strict_loading)

    data_path = os.path.join(args.data_path)
    image_path = os.path.join(args.data_path, "images")
    if hasattr(model, "image_transform"):
        transform = model.image_transform
    else:
        transform = get_default_image_transform(
            clip="clip" in cfg.MODEL.COMPOSITION)
    datasets = CFQSet(data_path, image_path, transform)

    print("Computing metrics...")
    results = datasets.compute_metrics(model, with_dots=False)
    primary = datasets.get_primary_metrics(results)
    for key, val in results.items():
        for kk, vv in val.items():
            if not isinstance(vv, dict):
                results[key][kk] = vv.to_dict()  # convert Series for json serialization

    for met, res in primary.items():
        print(met, res)

    if args.output_path is None:
        output_path = os.path.dirname(args.config_file)
        output_path = os.path.join(output_path, "cfq_results.json")
    else:
        output_path = args.output_path

    if args.debug:
        import IPython
        IPython.embed()

    with open(output_path, "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    evaluate()
