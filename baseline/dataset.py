import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

import albumentations
import albumentations.pytorch
import pandas as pd
import random
from itertools import chain
from glob import glob
from sklearn.model_selection import train_test_split
from facenet_pytorch import MTCNN

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def get_train_image_path(folder, path):
    ''' 각 이미지의 full_path를 얻는 함수

        folder(str) : 폴더 이름
        path(str) : train_image 폴더들의 상위 폴더 path, Default 설정해놨음

    '''

    file_path_list = glob(path + folder + '/*')
    return file_path_list


def figure_out_mask_label(file_path):
    ''' 마스크 착용 여부를 얻어내는 함수

        file_path(str) : file의 전체 경로 ex) 
            ex) ./input/data/train/images/000001_female_Asian ~~ /normal.jpg

    '''

    file_name = file_path.split('/')[-1]
    if 'incorrect' in file_name:
        return 'incorrect'
    elif 'mask' in file_name:
        return 'wear'
    else:
        return 'not_wear'


def get_label(label_dict, mask, gender, age):
    ''' label을 얻을 수 있는 함수

        label_dict(dict) : label값들을 가진 dictionary
        mask(str) : 마스크 착용 여부
        gender(str) : 성별
        age(int) : 나이

    '''

    if age < 30:
        age = 'young'
    elif (age >= 30 and age < 60):
        age = 'middle'
    else:
        age = 'old'

    key = '_'.join([mask, gender, age])
    return label_dict[key]


def get_folder_path(full_path_list):
    ''' stratification함수에서 폴더명을 뽑기 위해 필요한 함수

        full_path_list(list): 이미지 경로가 담긴 리스트

    '''

    folder_path_list = []
    for full_path in full_path_list:
        folder_path = full_path.split('/')[-2]  # 폴더명만 추출
        folder_path_list.append(folder_path)

    return folder_path_list


def stratification(df, label_count, infrequent_classes, ratio=0.2):
    '''
        df : label값을 구하고 파일 기준으로 분류된 df
        infrequent_classes : 숫자가 적은 class 번호 순으로 정렬된 list
        ratio : 얻고자 하는 validation ratio
    '''

    total_valid_count = int(len(df) * ratio / 7)  # valid용 folder의 개수
    valid_folder_list = []  # 여기에 valid용 folder명을 그룹마다 담을겁니다.
    count_summation = 0    # count_summation

    for class_num in infrequent_classes:
        # 만약 class_num이 마지막 infrequent_classes의 원소라면
        # total_valid_count를 맞추기 위해 그동안 쌓은 count_summation의 차만큼 뽑습니다.
        # why? 반올림으로 인해 완전히 나눠 떨어지지 않을 수도 있기 때문에
        if class_num == infrequent_classes[-1]:
            group_count = total_valid_count - count_summation
        else:
            group_count = round(label_count[class_num] * ratio)

        random.seed(42)  # 복원을 위해 seed 설정
        group_df = df[df['label'] == class_num]  # 현재 class_num을 가진 rows 추출
        # 현재 group에서 뽑아야 하는 개수만큼 sampling
        index = random.sample(list(group_df.index), group_count)
        # index들의 full_path를 얻은 후
        group_full_path = df.iloc[index]['full_path'].values
        group_folder_path = get_folder_path(
            group_full_path)  # folder명만 추출 (리스트)
        valid_folder_list.append(group_folder_path)  # valid_folder_list에 담고
        count_summation += group_count  # group_count를 쌓아간다.

    return valid_folder_list


def make_dataframe(train_data_dir, data_dir, ratio=0.2):
    train_df = pd.read_csv(os.path.join(train_data_dir, 'train.csv'))
    submission_df = pd.read_csv(os.path.join(train_data_dir, 'info.csv'))

    # 각 폴더 내에 있는 파일 경로 읽어오기
    # path_list는 array 형식으로 해당 폴더의 파일 경로 7개가 들어있음
    train_df['path_list'] = train_df['path'].apply(
        lambda x: get_train_image_path(x, path=(data_dir + '/')))

    # 리스트화된 컬럼을 gender, age, path에 맞게 펼쳐준 뒤, merge하여 새로운 df 생성
    gender_df = pd.DataFrame({'gender': np.repeat(train_df['gender'].values, train_df['path_list'].str.len()),
                              'full_path': np.concatenate(train_df['path_list'].values)
                              })

    age_df = pd.DataFrame({'age': np.repeat(train_df['age'].values, train_df['path_list'].str.len()),
                           'full_path': np.concatenate(train_df['path_list'].values)
                           })

    # 기존 DF의 path column의 이름을 folder로 변환
    path_df = pd.DataFrame({'folder': np.repeat(train_df['path'].values, train_df['path_list'].str.len()),
                            'full_path': np.concatenate(train_df['path_list'].values)
                            })
    # merge
    new_df = pd.merge(gender_df, age_df, how='inner', on='full_path')
    new_df = pd.merge(new_df, path_df, how='inner', on='full_path')
    # label Dictaionary는 라벨링을 할 때 사용됩니다.
    label_dict = defaultdict(list)

    label = 0
    for mask in ('wear', 'incorrect', 'not_wear'):
        for gender in ('male', 'female'):
            for age in ('young', 'middle', 'old'):
                key = '_'.join([mask, gender, age])
                label_dict[key] = label
                label += 1
    # 각 row마다 mask 여부 확인 후 mask column 생성
    # incorrect, wear, not_wear
    new_df['mask'] = new_df['full_path'].apply(figure_out_mask_label)

    # label 생성
    # mask, gender, age 조합을 가지고 각 row마다 label 생성
    new_df['label'] = new_df[['mask', 'gender', 'age']].apply(
        lambda x: get_label(label_dict, x[0], x[1], x[2]), axis=1)
    # label을 count하고 오름차순 정렬

    label_count = new_df['label'].value_counts().sort_values()

    # 마스크 오착용 or 마스크 미착용 label들을 가지고
    # 순차적으로 드문 라벨의 인덱스를 추출
    incorrect_classes = [6, 7, 8, 9, 10, 11]
    infrequent_classes = label_count[label_count.index.isin(
        incorrect_classes)].index

    # 설정한 ratio대로 train, valid split
    valid_folder_list = stratification(
        new_df, label_count, infrequent_classes, ratio)

    # 함수로 얻어낸 valid_folder_path는 2D-array형식이며
    # infrequent_classes개수에 맞게 6개 그룹으로 되어있음
    # ex) [[classes1_folders], [classes2_folders], ... [classes6_folders]]

    # 그러므로 작업 편의를 위해 1D-array로 변환
    valid_folder_list = list(chain(*valid_folder_list))
    # valid_df 생성
    # new_df의 folder명이 valid_folder_path에 해당하면 추출
    valid_df = new_df[new_df['folder'].isin(valid_folder_list)]

    # trainset 분리 위해 valid_df의 index 추출
    valid_index = valid_df.index

    # train_df 생성
    # new_df의 인덱스 중 valid_df의 인덱스가 아니면 train_df
    train_index = [idx for idx in new_df.index if idx not in valid_index]
    train_df = new_df.iloc[train_index]

    #print(len(train_df), len(valid_df))

    # path는 본인이 원하는 위치에 지정해주세요
    train_df.to_csv(os.path.join(
        train_data_dir, 'train_list.csv'), index=False)
    valid_df.to_csv(os.path.join(
        train_data_dir, 'valid_list.csv'), index=False)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(
                f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                # (resized_data, 000004_male_Asian_54, mask1.jpg)
                img_path = os.path.join(self.data_dir, profile, file_name)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print(
                "[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(
            mask_label, gender_label, age_label)

        image_transform = self.transform(image=np.array(image))['image']
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class TrainDataset_Crop(Dataset):
    def __init__(self, path, label, transform):
        img_list = []
        for p in path:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 얼굴 탐색
            mtcnn = MTCNN()
            boxes, prob = mtcnn.detect(img)

            # 얼굴을 찾지 못하는 경우
            if not isinstance(boxes, np.ndarray):
                img = img[110:382, 90:287]

            # 얼굴을 찾은 경우
            else:
                xmin, ymin, xmax, ymax = map(int, boxes[0])
                # 표준편차로 얼굴 주변까지 일부 확보
                xmin, ymin, xmax, ymax = abs(
                    xmin)-19, abs(ymin)-35, abs(xmax)+20, abs(ymax)+38
                if xmin <= 0:
                    xmin = 0
                if ymin <= 0:
                    ymin = 0
                img = img[ymin:ymax, xmin:xmax]

            img_list.append(img)

        self.X = img_list
        self.y = label
        self.transform = transform

    def __len__(self):
        len_dataset = len(self.X)
        return len_dataset

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        X = self.transform(image=X)['image']
        return X, y


class TrainDataset(Dataset):
    def __init__(self, path, label, transform):
        img_list = []
        for p in path:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img)

        self.X = img_list
        self.y = label
        self.transform = transform

    def __len__(self):
        len_dataset = len(self.X)
        return len_dataset

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        X = self.transform(image=X)['image']
        return X, y


def get_transforms(model):
    transformations = {}
    if model == 'model1':
        transformations['train'] = albumentations.Compose([albumentations.Resize(336, 336),
                                                           albumentations.OneOf(
                                                               [albumentations.GaussNoise()], p=0.2),
                                                           albumentations.OneOf([albumentations.MotionBlur(p=.2),
                                                                                 albumentations.MedianBlur(
                                                               blur_limit=3, p=0.1),
                                                               albumentations.Blur(blur_limit=3, p=0.1)], p=0.2),
                                                           albumentations.ShiftScaleRotate(
                                                               shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                                                           albumentations.OneOf([albumentations.OpticalDistortion(p=0.3),
                                                                                 albumentations.GridDistortion(p=.1)], p=0.2),
                                                           albumentations.OneOf([albumentations.CLAHE(clip_limit=2),
                                                                                 albumentations.Sharpen(),
                                                                                 albumentations.Emboss(),
                                                                                 albumentations.RandomBrightnessContrast()], p=0.3),
                                                           albumentations.HueSaturationValue(
                                                               p=0.3),
                                                           albumentations.Cutout(
                                                               num_holes=4, max_h_size=3, max_w_size=3, fill_value=0, p=0.2),
                                                           albumentations.Normalize(
                                                               (0.548, 0.504, 0.479), (0.237, 0.247, 0.246)),
                                                           albumentations.pytorch.transforms.ToTensorV2()])
        transformations['val'] = albumentations.Compose([albumentations.Resize(336, 336),
                                                         albumentations.Normalize(
                                                             (0.548, 0.504, 0.479), (0.237, 0.247, 0.246)),
                                                         albumentations.pytorch.transforms.ToTensorV2()])
    if model == 'model2':
        transformations['train'] = albumentations.Compose([albumentations.Resize(600, 600),
                                                           # albumentations.RandomRotation(15),
                                                           albumentations.HorizontalFlip(
                                                               p=0.3),
                                                           albumentations.OneOf([albumentations.ShiftScaleRotate(rotate_limit=15, p=0.5),
                                                                                 albumentations.RandomBrightnessContrast(
                                                               brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                                                               albumentations.HorizontalFlip(
                                                                   p=0.5),
                                                               albumentations.MotionBlur(
                                                                   p=0.5),
                                                               albumentations.OpticalDistortion(
                                                                   p=0.5),
                                                               albumentations.GaussNoise(p=0.5)], p=1),
                                                           albumentations.Normalize(
                                                               (0.548, 0.504, 0.479), (0.237, 0.247, 0.246)),
                                                           albumentations.pytorch.transforms.ToTensorV2()])
        transformations['val'] = albumentations.Compose([albumentations.Resize(600, 600),
                                                         albumentations.Normalize(
                                                             (0.548, 0.504, 0.479), (0.237, 0.247, 0.246)),
                                                         albumentations.pytorch.transforms.ToTensorV2()])
    if model == 'model3':
        transformations['train'] = albumentations.Compose(
            [
                albumentations.Resize(456, 456),
                #       albumentations.RandomRotation(15),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.6),
                albumentations.RandomBrightnessContrast(
                    brightness_limit=0.1, p=0.6),
                albumentations.GaussNoise(p=0.5),
                albumentations.MotionBlur(p=0.5),
                albumentations.OpticalDistortion(
                    p=0.5),
                albumentations.Normalize(
                    (0.548, 0.504, 0.479), (0.237, 0.247, 0.246)),
                albumentations.pytorch.transforms.ToTensorV2(),
                #       이미지 원본 사이즈는 384, 512
            ]
        )
        transformations['val'] = albumentations.Compose(
            [
                albumentations.Resize(456, 456),
                albumentations.Normalize(
                    (0.548, 0.504, 0.479), (0.237, 0.247, 0.246)),
                albumentations.pytorch.transforms.ToTensorV2()
                #       이미지 원본 사이즈는 384, 512
            ]
        )
    if model == 'model4':
        img_size = (512, 384)
        transformations['train'] = albumentations.Compose([
            albumentations.Resize(img_size[0], img_size[1], p=1.0),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albumentations.GaussNoise(p=0.5),
            albumentations.Normalize(mean=(0.548, 0.504, 0.479), std=(
                0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
            albumentations.pytorch.transforms.ToTensorV2(),
        ], p=1.0)
        transformations['val'] = albumentations.Compose([
            albumentations.Resize(img_size[0], img_size[1]),
            albumentations.Normalize(mean=(0.548, 0.504, 0.479), std=(
                0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
            albumentations.pytorch.transforms.ToTensorV2(),
        ], p=1.0)

    if model == 'model5':
        img_size = (512, 384)
        transformations['train'] = albumentations.Compose([
            albumentations.Resize(img_size[0], img_size[1], p=1.0),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albumentations.GaussNoise(p=0.5),
            albumentations.Normalize(mean=(0.548, 0.504, 0.479), std=(
                0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
            albumentations.pytorch.transforms.ToTensorV2(),
        ], p=1.0)
        transformations['val'] = albumentations.Compose([
            albumentations.Resize(img_size[0], img_size[1]),
            albumentations.Normalize(mean=(0.548, 0.504, 0.479), std=(
                0.237, 0.247, 0.246), max_pixel_value=255.0, p=1.0),
            albumentations.pytorch.transforms.ToTensorV2(),
        ], p=1.0)
    return transformations


_dataset_entrypoints = {
    'model1': TrainDataset_Crop,
    'model2': TrainDataset,
    'model3': TrainDataset_Crop,
    'model4': MaskBaseDataset,
    'model5': TrainDataset
}


def dataset_entrypoint(dataset_name):
    return _dataset_entrypoints[dataset_name]


def is_dataset(dataset_name):
    return dataset_name in _dataset_entrypoints


def create_dataset(dataset_name, **kwargs):
    if is_dataset(dataset_name):
        create_fn = dataset_entrypoint(dataset_name)
        dataset = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown dataset (%s)' % dataset_name)
    return dataset


class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
