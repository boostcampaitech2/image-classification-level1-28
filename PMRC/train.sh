#!/bin/bash

data_dir='/tf/pstage_1/train/images';
save_dir='test';
p_rotate=0.5
p_contrast=0.5
model='EfficientNetB4'
criterion='focal'
optimizer='AdamW'
lr=3e-4
lr_decay_step=2
batch_size=32
dataset='TrainDataset'

#SM_CHANNEL_TRAIN=$data_dir SM_MODEL_DIR=$save_dir python train.py
SM_CHANNEL_TRAIN=$data_dir SM_MODEL_DIR=$save_dir python train_check.py --epochs 15 \
	--augmentation CustomAugmentation --p_rotate $p_rotate --p_contrast $p_contrast \
	--model $model --criterion $criterion --optimizer $optimizer --lr $lr --lr_decay_step $lr_decay_step \
	--batch_size $batch_size --dataset $dataset
