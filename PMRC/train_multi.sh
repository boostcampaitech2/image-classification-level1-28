#!/bin/bash

data_dir='/tf/pstage_1/train/images';
save_dir='model_save';
name='EfficientB4'
p_rotate=0.5
p_contrast=0.5
model='EfficientNetB4'
criterion='focal'
optimizer='AdamW'
lr=3e-4
lr_decay_step=2
batch_size=32
dataset='TrainDataset'

seed=("42" "24")
n_seed=${#seed[@]}

for (( idx_seed=0; idx_seed<${n_seed}; idx_seed++));
do
  echo ${seed[$idx_seed]}

  SM_CHANNEL_TRAIN=$data_dir SM_MODEL_DIR=$save_dir python train_check.py --epochs 1 \
	--augmentation CustomAugmentation --p_rotate $p_rotate --p_contrast $p_contrast \
	--model $model --criterion $criterion --optimizer $optimizer --lr $lr --lr_decay_step $lr_decay_step \
	--batch_size $batch_size --dataset $dataset --seed ${seed[$idx_seed]} --name $name-${seed[$idx_seed]} \
	> $name-${seed[$idx_seed]}-log.txt
done



