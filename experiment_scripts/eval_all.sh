
for expdir in /home/default/ephemeral_drive/experiments/paper/*; do
  [ -e "${expdir}/config.yaml" ] || continue  # skip empty or empty glob
  [ ! -e "${expdir}/claimed.temp" ] || continue  # skip what another process is doing
  echo ${expdir}
  touch ${expdir}/claimed.temp
  bash experiment_scripts/eval.sh $expdir
  rm ${expdir}/claimed.temp
done
