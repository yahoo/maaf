# Copyright 2020 Verizon Media, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

python main.py \
    --dataset=mitstates \
    --dataset_path=/home/default/ephemeral_drive/datasets/mitstates/ \
    --num_iters=150000 \
    --model=sequence_concat_attention \
    --loss=batch_based_classification \
    --savedir=/home/default/ephemeral_drive/experiments/mitstates/ \
    --image_model_arch=resnet50 \
    --learning_rate_decay_frequency=50000 \
    --exp_name=sequence_concat_attention_34_1_32_512_1_1_1_50000_0 \
    --text_model_arch=lstm \
    --pretrained_weight_lr_factor_text=1.0 \
    --pretrained_weight_lr_factor_image=0.1 \
    --att_layer_spec=34 \
    --number_attention_blocks=1 \
    --width_per_attention_block=32 \
    --embed_dim=512 \
    --sequence_concat_img_through_attn \
    --sequence_concat_include_text \
    --resolutionwise_pool
