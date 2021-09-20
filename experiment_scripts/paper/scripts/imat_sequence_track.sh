# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


# set these
TAG="crm-imat"
pretrain_config=experiment_scripts/paper/base_configs/clipresmaaf_imat.yaml
OUTPUT_DIR=/home/default/ephemeral_drive/experiments/track/

# derived constants
EXP_NAME=${TAG}_$(date "+%Y-%m-%d-%H-%M-%S")
pretrain_dir=${OUTPUT_DIR}/$EXP_NAME
finetune_dir=${pretrain_dir}-finetune

# pretrain
python main.py --config $pretrain_config --no-timestamp \
  OUTPUT_DIR $OUTPUT_DIR EXP_NAME ${EXP_NAME}_0 \
  SOLVER.NUM_ITERS 7254
for i in 1 2 3 4 5 6; do
  j=$(($i - 1))
  bash experiment_scripts/eval.sh ${pretrain_dir}_$j
  python main.py --config $pretrain_config --no-timestamp \
    OUTPUT_DIR $OUTPUT_DIR EXP_NAME ${EXP_NAME}_$i \
    SOLVER.NUM_ITERS 7254 \
    MODEL.WEIGHTS ${pretrain_dir}_${j}/latest_checkpoint.pth
done

bash experiment_scripts/eval.sh ${pretrain_dir}_6

# fine tune (and eval) on Fashion IQ
python main.py --config $pretrain_dir/config.yaml --no-timestamp \
  EXP_NAME ${EXP_NAME}_6-finetune \
  DATASET.NAME fashioniq \
  DATASET.PATH /home/default/ephemeral_drive/Data/FashionIQ/ \
  DATASET.AUGMENTATION.IMAGE_AUGMENTATION True \
  SOLVER.LEARNING_RATE_DECAY_FREQUENCY 4000 \
  SOLVER.NUM_ITERS 6000 \
  MODEL.WEIGHTS ${pretrain_dir}_6/latest_checkpoint.pth

bash experiment_scripts/eval.sh ${pretrain_dir}_6-finetune
