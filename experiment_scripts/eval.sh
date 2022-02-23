# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.


expdir=$1

if [[ $expdir == *"baseline"* ]] && [[ ! $expdir == *"finetune" ]]; then
  weights=None
else
  weights=$expdir/latest_checkpoint.pth
fi


if ! compgen -G "${expdir}/fashioniq*eval.json" > /dev/null || [ $REDO_EVALS ]; then
  python main.py --config $expdir/config.yaml \
    --no-train --no-timestamp --no-config-save \
    DATASET.NAME fashioniq \
    DATASET.PATH /home/default/ephemeral_drive/Data/FashionIQ/ \
    MODEL.WEIGHTS $weights
fi
if ! compgen -G "${expdir}/imat_fashion*eval.json" > /dev/null || [ $REDO_EVALS ]; then
  python main.py --config $expdir/config.yaml \
    --no-train --no-timestamp --no-config-save \
    DATASET.NAME imat_fashion \
    DATASET.PATH /home/default/ephemeral_drive/Data/imat2018/ \
    MODEL.WEIGHTS $weights \
    DATASET.AUGMENTATION.IMAGE_AUGMENTATION None
fi
if ! compgen -G "${expdir}/fashiongen*eval.json" > /dev/null || [ $REDO_EVALS ]; then
  python main.py --config $expdir/config.yaml \
    --no-train --no-timestamp --no-config-save \
    DATASET.NAME fashiongen \
    DATASET.PATH /home/default/ephemeral_drive/Data/fashiongen/ \
    MODEL.WEIGHTS $weights \
    DATASET.AUGMENTATION.IMAGE_AUGMENTATION None
fi
if [[ ! -e $expdir/cfq_results.json ]] || [ $REDO_EVALS ]; then
  python src/maaf/actions/eval_cfq.py --config $expdir/config.yaml
fi
