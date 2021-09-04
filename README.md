# README

## Mask Image Classification

Code for solution in Mask Image Classification hosted by Naver Boostcamp AI Tech.

To learn more detail about the competition, please, refer to [**the AI Stage post**](https://stages.ai/competitions/74/overview/description)

## **Archive contents**

`minibatch28/
├── data/
|   ├── image/
|   |   ├── train/ 
│   |   |   ├── 00001_male_Asian_40/
│   |   |   |   ├── mask.jpg
│   |   |   |   ├── mask2.jpg
│   |   |   |   ├── incorrect.png 
│   |   |   |   └── normal.jpeg 
│   |   |   ├── {Number}_{Gender}_{Race}_{Age}/
│   |   |   |   ├── ...
│   |   |   |   └── ...
|   |   |   └── 99999_female_Asian_150/
│   |   |       ├── mask.jpg 
│   |   |       ├── incorrect.jpg
│   |   |       └── normal.jpg 
|   |   └── eval/ 
│   |       ├── abcde.jpg
│   |       ├── {Any_image_name}.jpg
│   |       └── lorem_ipsum.jpeg 
│   ├── train.csv
│   └── info.csv
├── output/
│   └── ensemble/ 
├── models/
├── dataset.py
├── loss.py
├── inference.py
├── train.py
└── train.sh`

- `data/` : contains raw data dir and label data (should contain 'train.csv', 'info.csv')
- `data/image/` : raw image dir of the competition
- `data/eval/` : evaluation image dir of the competition
- `output/` : inference result csv files will be created
- `output/ensemble/` : ensemble result csv files will be created
- `models/` : contains trained state_dict of each model

### **Requirements**

- Ubuntu 18.04.5 LTS
- Python 3.8.5
- Pytorch 1.7.1
- CUDA 11.0

You can use the `pip install -r requirements.txt` to install the necessary packages.

### **Hardware**

- CPU: 8 x Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- GPU: 1 x Tesla V-100
- RAM: 88G

## **Prepare Data**

You can automatically generate `data/train_list.csv`, `data/valid_list.csv` files by running `train.py`

## **Train Model**

To train model, run following command.

`$ python train.py \`

    `--model {model_number} --dataset {model_number} --batch_size {batch_size} --epochs {epochs} \`

    `--lr_decay_step {lr_decay_step} --gamma {gamma} --lr {learning_rate} --scheduler 1 \` 

    `--cutmix 0 --criterion {model_number} --optimizer {optimizer}`

To train 5 models at once, run following shell script file.

`$ ./train.sh`

## **Predict**

If trained weights are prepared, you can create files that contains class of images.

`$ python inference.py --model {model_number} --batch_size {batch_size}` 

## Ensemble

If you want to do an ensemble, add `--ensemble` argument like following code.

`$ python inference.py --model {model_number} --batch_size {batch_size} --ensemble 1`

It will make csv files for ensemble in dir `./output/` 

,and by following command to create ensembled result `$ python ensemble.py`

Then `ensemble.csv` will be created in `output/ensemble` directory.

### Member of Minibatch 28

-