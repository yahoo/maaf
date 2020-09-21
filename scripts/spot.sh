# Copyright 2020 Verizon Media, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

python main.py --dataset=spotthediff \
  --dataset_path=/home/edodds/Data/spot-the-diff/data/ \
  --num_iters=90000 --model=sequence_concat_attention  \
  --loss=batch_based_classification \
  --savedir=/home/edodds/experiments/spot/ \
  --image_model_arch=resnet50 \
  --learning_rate_decay_frequency=30000 \
  --exp_name=spot_seqcat \
  --text_model_arch=lstm \
  --pretrained_weight_lr_factor_text=1.0 \
  --pretrained_weight_lr_factor_image=0.1 \
  --att_layer_spec=34 \
  --number_attention_blocks=2 \
  --sequence_concat_img_through_attn \
  --sequence_concat_include_text \
  --resolutionwise_pool \
  --eval_every=1 \
  --loader_num_workers=0
