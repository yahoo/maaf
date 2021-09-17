# set these
TAG="clipresmaaf-scramble-baseline-14k"
pretrain_config=experiment_scripts/paper/base_configs/clipresmaaf_fiq.yaml
OUTPUT_DIR=/home/default/ephemeral_drive/experiments/paper/

# derived constants
EXP_NAME=${TAG}_$(date "+%Y-%m-%d-%H-%M-%S")
pretrain_dir=${OUTPUT_DIR}/$EXP_NAME
finetune_dir=${pretrain_dir}-finetune


# eval baseline
python main.py --config $pretrain_config --no-train --no-timestamp \
  MODEL.CLIP.MISALIGNMENT scramble \
  OUTPUT_DIR $OUTPUT_DIR EXP_NAME $EXP_NAME
bash experiment_scripts/eval.sh $pretrain_dir

# fine tune (and eval) on Fashion IQ
python main.py --config $pretrain_dir/config.yaml --no-timestamp \
  EXP_NAME ${EXP_NAME}-finetune \
  DATASET.NAME fashioniq \
  DATASET.PATH /home/default/ephemeral_drive/Data/FashionIQ/ \
  DATASET.AUGMENTATION.IMAGE_AUGMENTATION True \
  MODEL.CLIP.MISALIGNMENT scramble \
  SOLVER.NUM_ITERS 14000 \
  SOLVER.LEARNING_RATE_DECAY_FREQUENCY 10000

# eval others with fiq-tuned model
bash experiment_scripts/eval.sh $finetune_dir
