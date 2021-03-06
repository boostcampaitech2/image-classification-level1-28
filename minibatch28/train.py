import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import pandas as pd
from dataset import MaskBaseDataset, create_dataset, get_transforms, make_dataframe
from loss import create_criterion
from model import create_model
from sklearn.metrics import f1_score

import wandb


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # 패치의 중앙 좌표 값 cx, cy
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 모서리 좌표 값
    bbx1 = 0
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = W
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    print('train start')

    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.model))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    patience = args.patience
    counter = 0

    # -- dataset
    df_train = pd.read_csv(os.path.join(
        args.train_data_dir, 'train_list.csv'))
    df_val = pd.read_csv(os.path.join(
        args.train_data_dir, 'valid_list.csv'))
    if args.dataset == 'model4':
        dataset = MaskBaseDataset(data_dir)
        train_set, val_set = dataset.split_dataset()
        train_set.dataset.set_transform(get_transforms(args.model)['train'])
        val_set.dataset.set_transform(get_transforms(args.model)['val'])
    else:
        train_set = create_dataset(
            args.dataset,
            path=df_train['full_path'].values,
            label=df_train['label'].values,
            transform=get_transforms(args.model)['train']
        )
        val_set = create_dataset(
            args.dataset,
            path=df_val['full_path'].values,
            label=df_val['label'].values,
            transform=get_transforms(args.model)['val']
        )
    print('dataset created')

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    print('data loaded')
    # -- model
    model = create_model(args.model).to(device)
    model = torch.nn.DataParallel(model)

    wandb.watch(model)
    os.makedirs(save_dir, exist_ok=True)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"),
                         args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    if args.scheduler:
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=args.gamma)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        print('epoch', epoch+1)
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            optimizer.zero_grad()

            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            if args.cutmix and (args.beta > 0 and np.random.random() > 0.5):  # cutmix가 실행될 경우
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(inputs.size()[0]).to(device)
                target_a = labels  # 원본 이미지 label
                target_b = labels[rand_index]  # 패치 이미지 label
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index,
                                                            :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                           (inputs.size()[-1] * inputs.size()[-2]))
                logits = model(inputs)
                loss = criterion(logits, target_a) * lam + \
                    criterion(logits, target_b) * (1. - lam)
                loss_value += loss.item()
                if (idx + 1) % args.log_interval == 0:
                    print("Cutmix")
            else:
                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)
                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    wandb.log({"Train loss": train_loss,
                              "Train Acc": train_acc,
                               "Learning Rate": current_lr})

                    loss_value = 0
                    matches = 0

            loss.backward()
            optimizer.step()

        if args.scheduler:
            scheduler.step()

        # val loop
        print("Start Val")
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            example_images = []

            for idx, val_batch in enumerate(val_loader):
                inputs, labels = val_batch
                if (idx % 500) == 0:
                    example_images.append(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).detach().item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)

            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
                counter = 0
            else:
                counter += 1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            if counter > patience:
                print("Early Stopping...")
                break
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            f1_score_value = f1_score(labels.detach().cpu().numpy(
            ), preds.detach().cpu().numpy(), average='macro')
            print(f"f1-score : {f1_score_value:.3f}")

            wandb.log(
                {"Validation Loss": val_loss,
                 "Validation Acc": val_acc,
                 "Validation F1": f1_score_value,
                 "example_images": [wandb.Image(image) for image in example_images]})
            print()


if __name__ == '__main__':
    # wandb.login()
    wandb.init(entity="minibatch28",
               project="MaskClassification", name="minibatch28_final")

    parser = argparse.ArgumentParser()

    #from dotenv import load_dotenv
    import os
    # load_dotenv(verbose=True)

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str,
                        help='dataset augmentation type')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default='BaseModel',
                        help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer type (default: SGD)')
    parser.add_argument('--scheduler', type=int, default=1,
                        help='scheduler On/Off (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20,
                        help='learning rate scheduler decay step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp',
                        help='model save at {name}')
    parser.add_argument('--patience', type=int, default=100,
                        help='early stopping boundary')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma for LR scheduler')
    parser.add_argument('--cutmix', type=int, default=0,
                        help='Whether to use cutmix (default: 0)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta for cutmix (default: 1.0)')

    # Container environment
    parser.add_argument('--data_dir', type=str,
                        default='./data/image/train')
    parser.add_argument('--eval_dir', type=str,
                        default='./data/image/eval')
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--train_data_dir', type=str, default='./data')

    args = parser.parse_args()
    wandb.config.update(args)
    print(args)

    train_data_dir = args.train_data_dir
    data_dir = args.data_dir
    model_dir = args.model_dir

    make_dataframe(train_data_dir, data_dir)
    train(data_dir, model_dir, args)
