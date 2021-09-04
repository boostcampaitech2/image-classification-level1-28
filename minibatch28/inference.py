import argparse
import os
from importlib import import_module
import torch.nn.functional as F
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from model import create_model


def load_model(saved_model, device, args):
    model = create_model(args.model)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def get_size(model):
    size = {
        'model1': (336, 336),
        'model2': (600, 600),
        'model3': (456, 456),
        'model4': (512, 384),
        'model5': (512, 384)
    }
    return size[model]


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    ensemble = args.ensemble
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 18
    model = load_model(model_dir, device, args).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    ###############
    info = pd.read_csv(info_path).iloc[:100, :]
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, get_size(args.model))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            if ensemble:
                pred = F.softmax(pred, dim=1)
            else:
                pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())
    if ensemble:
        df = pd.DataFrame(preds)
        merge = pd.merge(info, df, left_index=True, right_index=True)
        merge = merge.drop(['ans'], axis=1)
        merge.to_csv(os.path.join(
            output_dir, '{}.csv'.format(args.model)), index=False)
    else:
        info['ans'] = preds
        info.to_csv(os.path.join(
            output_dir, '{}.csv'.format(args.model)), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel',
                        help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str,
                        default='/opt/ml/input/data/eval')
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--ensemble', type=int, default=0)

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = os.path.join(args.model_dir, args.model)
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
