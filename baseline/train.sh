python train.py --model model1 --dataset model1 --batch_size 20 --epochs 11 --lr_decay_step 3 --gamma 0.3 --lr 3e-5 --scheduler 1 --cutmix 0 --criterion model1 --optimizer AdamW

python train.py --model model2 --dataset model2 --batch_size 7 --epochs 15 --lr_decay_step 2 --gamma 0.3 --lr 3e-5 --scheduler 1 --cutmix 0 --criterion model2 --optimizer AdamW

python train.py --model model3 --dataset model3 --batch_size 16 --epochs 8 --lr_decay_step 1 --gamma 0.85 --lr 3e-5 --scheduler 1 --cutmix 1 --criterion model3 --optimizer Adam

python train.py --model model4 --dataset model4 --batch_size 32 --epochs 15 --lr 1e-4 --scheduler 0 --cutmix 0 --criterion model4 --optimizer Adam

python train.py --model model5 --dataset model5 --batch_size 32 --epochs 15 --lr 1e-4 --scheduler 0 --cutmix 0 --criterion model5 --optimizer Adam