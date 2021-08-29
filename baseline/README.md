# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.8.0+cu111
- torchvision==0.9.0+cu111

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN=[train image dir] SM_MODEL_DIR=[model saving dir] python train.py`
- `python train.py`로 실행 시 학습 데이터 디렉토리 위치는 `/opt/ml/input/data/train/images`, 모델이 저장될 디렉토리 위치는 `./model` 입니다.
- 학습 시, option을 argument로 지정할 수 있습니다. ex) `python train.py --ephochs 10 --optimizer Adam`
- `train.py`에서 `wandb.init()`의 `name`을 변경 후 실행해주세요.

### Inference
- `SM_CHANNEL_EVAL=[eval image dir] SM_CHANNEL_MODEL=[model saved dir] SM_OUTPUT_DATA_DIR=[inference output dir] python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR=[GT dir] SM_OUTPUT_DATA_DIR=[inference output dir] python evaluation.py`
