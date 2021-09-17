# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

python main.py --dataset=birds \
  --dataset_path=/home/default/Data/birds-to-words/ \
  --num_iters=50000 --model=sequence_concat_attention  \
  --loss=batch_based_classification \
  --savedir=/home/edodds/experiments/birds/ \
  --image_model_arch=resnet50 \
  --learning_rate_decay_frequency=10000 \
  --exp_name=birds_seqcat_quicker \
  --text_model_arch=lstm \
  --pretrained_weight_lr_factor_text=1.0 \
  --pretrained_weight_lr_factor_image=0.1 \
  --att_layer_spec=34 \
  --number_attention_blocks=2 \
  --sequence_concat_img_through_attn \
  --sequence_concat_include_text \
  --resolutionwise_pool \
  --eval_every=5 \
