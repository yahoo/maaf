# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


for expdir in /home/default/ephemeral_drive/experiments/paper/*; do
  [ -e "${expdir}/config.yaml" ] || continue  # skip empty or empty glob
  [ ! -e "${expdir}/claimed.temp" ] || continue  # skip what another process is doing
  echo ${expdir}
  touch ${expdir}/claimed.temp
  bash experiment_scripts/eval.sh $expdir
  rm ${expdir}/claimed.temp
done
