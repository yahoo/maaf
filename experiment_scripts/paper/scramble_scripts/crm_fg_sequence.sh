# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


# set these
TAG="clip-scramble-resmaaf01-fg"
pretrain_config=experiment_scripts/paper/base_configs/clipresmaaf_fg.yaml
OUTPUT_DIR=/home/default/ephemeral_drive/experiments/paper/

# derived constants
EXP_NAME=${TAG}_$(date "+%Y-%m-%d-%H-%M-%S")
pretrain_dir=${OUTPUT_DIR}/$EXP_NAME
finetune_dir=${pretrain_dir}-finetune

# pretrain
python main.py --config $pretrain_config --no-timestamp \
  OUTPUT_DIR $OUTPUT_DIR EXP_NAME $EXP_NAME \
  MODEL.CLIP.MISALIGNMENT scramble

# eval after only pretraining
bash experiment_scripts/eval.sh $pretrain_dir

# fine tune (and eval) on Fashion IQ
python main.py --config $pretrain_dir/config.yaml --no-timestamp \
  EXP_NAME ${EXP_NAME}-finetune \
  DATASET.NAME fashioniq \
  DATASET.PATH /home/default/ephemeral_drive/Data/FashionIQ/ \
  DATASET.AUGMENTATION.IMAGE_AUGMENTATION True \
  SOLVER.LEARNING_RATE_DECAY_FREQUENCY 4000 \
  SOLVER.NUM_ITERS 6000 \
  MODEL.WEIGHTS $pretrain_dir/latest_checkpoint.pth

# eval others with fiq-tuned model
bash experiment_scripts/eval.sh $finetune_dir
