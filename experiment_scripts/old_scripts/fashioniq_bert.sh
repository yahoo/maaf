# Copyright 2022 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

python main.py \
    --dataset=fashioniq \
    --dataset_path=/home/default/Data/FashionIQ/ \
    --num_iters=150000 \
    --model=sequence_concat_attention \
    --loss=batch_based_classification \
    --savedir=/home/default/ephemeral_drive/experiments/fashioniq/ \
    --image_model_arch=resnet50 \
    --learning_rate_decay_frequency=50000 \
    --exp_name=sequence_concat_attention_34_2_128_1_1_1_0 \
    --text_model_arch=lstm \
    --pretrained_weight_lr_factor_text=1.0 \
    --pretrained_weight_lr_factor_image=0.1 \
    --att_layer_spec=34 \
    --number_attention_blocks=2 \
    --width_per_attention_block=128 \
    --sequence_concat_img_through_attn \
    --sequence_concat_include_text \
    --resolutionwise_pool
