OUTPUT_DIR=/home/default/ephemeral_drive/experiments/afhp
BASE_CONFIG=experiment_scripts/extras/clipmaaf.yaml


TAG="clipmaaf"
EXP_NAME=${TAG}_$(date "+%Y-%m-%d-%H-%M-%S")
exp_dir=${OUTPUT_DIR}/$EXP_NAME

python main.py --config $BASE_CONFIG --no-timestamp \
  EXP_NAME $EXP_NAME \
  OUTPUT_DIR $OUTPUT_DIR

bash experiment_scripts/eval.sh $exp_dir
