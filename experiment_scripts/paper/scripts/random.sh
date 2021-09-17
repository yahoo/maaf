# set these
TAG="random-baseline"
pretrain_config=experiment_scripts/paper/base_configs/random.yaml
OUTPUT_DIR=/home/default/ephemeral_drive/experiments/paper/

# derived constants
EXP_NAME=${TAG}_$(date "+%Y-%m-%d-%H-%M-%S")
pretrain_dir=${OUTPUT_DIR}/$EXP_NAME


# eval baseline
python main.py --config $pretrain_config --no-train --no-timestamp \
  EXP_NAME $EXP_NAME
bash experiment_scripts/eval.sh $pretrain_dir
