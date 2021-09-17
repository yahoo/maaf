# Copyright 2021 Yahoo, Licensed under the terms of the Apache License, Version 2.0.
# See LICENSE file in project root for terms.

python main.py --dataset=css3d \
    --eval_every=10 \
    --dataset_path=/home/default/ephemeral_drive/datasets/css/ \
    --num_iters=60000 \
    --model=sequence_concat_attention \
    --loss=batch_based_classification \
    --savedir=/home/default/ephemeral_drive/experiments/css/ \
    --image_model_arch=resnet50 \
    --learning_rate_decay_frequency=50000 \
    --exp_name=sequence_concat_attention_4_1_256_512_1_1_1_50000_60000_0_resnet50 \
    --text_model_arch=lstm \
    --pretrained_weight_lr_factor_text=1.0 \
    --pretrained_weight_lr_factor_image=0.1 \
    --att_layer_spec=4 \
    --number_attention_blocks=1 \
    --width_per_attention_block=256 \
    --embed_dim=512 \
    --sequence_concat_img_through_attn \
    --sequence_concat_include_text \
    --resolutionwise_pool \
    --attn_softmax_replacement=identity
