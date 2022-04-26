OUTPUT_DIR=/home/default/ephemeral_drive/experiments/tirg
BASE_CONFIG=experiment_scripts/extras/cliptirg.yaml

TAG="ctirg-fiq"
EXP_NAME=${TAG}_$(date "+%Y-%m-%d-%H-%M-%S")
exp_dir=${OUTPUT_DIR}/$EXP_NAME

python main.py --config $BASE_CONFIG --no-timestamp \
  EXP_NAME $EXP_NAME \
  OUTPUT_DIR $OUTPUT_DIR \

bash experiment_scripts/eval.sh $exp_dir
