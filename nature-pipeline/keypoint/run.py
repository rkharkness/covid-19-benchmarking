#!/usr/bin/env python3
import fire
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fastai.metrics import add_metrics
from fastai.callback import Callback

import cv2
from PIL import Image

from sklearn.metrics.pairwise import paired_euclidean_distances
import dsntnn


def parse_device(device):
    if device is None:
        devices = []
    elif not ',' in device:
        devices = [device]
    else:
        devices = device.split(',')
    return devices


def image_verify(path):
    try:
        if not path.is_file():
            path = None
        else:
            Image.open(path).verify()
    except Exception as e:
        print(path)
        print(e)
        path = None
    return path


class KeypointEucMetric(Callback):
    def __init__(self):
        self.name = f'L2loss'


    def on_epoch_begin(self, **kwargs):
        self._label = []
        self._score = []
        

    def on_batch_end(self, last_output, last_target, **kwargs):
        last_output = last_output[0]
        self._label += [last_target.to('cpu').data.numpy()]
        self._score += [last_output.to('cpu').data.numpy()]    


    def on_epoch_end(self, last_metrics, **kwargs):
        label = np.concatenate(self._label)
        score = np.concatenate(self._score)
        label = label.reshape(-1,2)
        score = score.reshape(-1,2)
        euc_loss = paired_euclidean_distances(label,score).mean()/2
        return add_metrics(last_metrics, euc_loss)


class KeyPointLoss(nn.Module):
    def __init__(self):
        super(KeyPointLoss, self).__init__()


    def forward(self, input, target):
        coords, heatmaps = input
        euc_losses = dsntnn.euclidean_losses(coords, target)
        euc_losses = euc_losses / 2
        reg_losses = dsntnn.js_reg_losses(heatmaps, target, sigma_t=1.0)
        reg_losses = reg_losses / 2
        loss = dsntnn.average_loss(euc_losses + reg_losses)

        return loss


class VisionRunner(object):
    def __init__(self, lr=1e-3, valid_ratio=0.2, n_epoch=10, 
                    size=(224,224), batch_size=32, num_workers=16, device=None,
                    unfreeze=False, model='models.resnet18',
                    fp16=False,
                    ):
        super(VisionRunner, self).__init__()
        self._devices = parse_device(device)
        self._size = size
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._n_epoch = n_epoch
        self._lr = lr
        self._valid_ratio = valid_ratio

        self._model = model
        self._unfreeze = unfreeze
        self._fp16 = fp16


    def train(self, image_dir, df_path, output_dir, 
                        col_image='image_path', col_group=None):
        import matplotlib
        matplotlib.use('Agg')
        from fastai.vision import Learner
        from fastai.vision import DataBunch, DatasetType
        from fastai.callbacks import SaveModelCallback

        image_dir = Path(image_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        model_output_name = 'scripted_model.zip'

        lr = self._lr
        valid_ratio = self._valid_ratio
        size = self._size
        if not isinstance(size,int):
            size, _ = size
        num_workers = self._num_workers
        batch_size = self._batch_size
        n_epoch = self._n_epoch
        devices = self._devices
        print(devices)
        if len(devices) == 0 or devices[0].lower() != 'cpu':
            pin_memory = True
            device_data = devices[0]
        else:
            pin_memory = False
            device_data = None

        df = pd.read_csv(df_path)

        from dataset.keypoint_dataset import KeypointDataset
        df_train = df.sample(frac=1-valid_ratio)
        df_valid = df.drop(df_train.index)
        ds_train = KeypointDataset(df_train,image_dir,height=size,width=size, p_aug=0.5)
        ds_valid = KeypointDataset(df_valid,image_dir,height=size,width=size, p_aug=0.5)
        data = DataBunch.create(ds_train, ds_valid, bs=self._batch_size, 
            num_workers=num_workers, pin_memory=pin_memory,device=device_data)


        from models.keypoint import CoordLocationNetwork
        n_locations = ds_train.n_locations
        model_name = self._model
        print(model_name)
        model = CoordLocationNetwork(n_locations, size, model_name)
        
        loss_func = KeyPointLoss()
        metrics = []
        metrics += [KeypointEucMetric()]
        learn = Learner(data, model, metrics=metrics, wd=1e-2, path=output_dir, loss_func=loss_func)

        model_single = learn.model
        if len(devices) >= 2:
            learn.model = nn.DataParallel(model_single,device_ids=devices)

        learn.fit_one_cycle(n_epoch, slice(lr), pct_start=0.3, 
            callbacks=[SaveModelCallback(learn, every='improvement',monitor='valid_loss', name='best')])

        preds_test = []
        for x, y in tqdm(DataLoader(ds_test, batch_size=batch_size),leave=False):
            if device_data:
                x = x.to(device_data)
            with torch.no_grad():
                coords, heatmaps = learn.model(x)
                coords = (coords+1)/2
                coords = coords.to('cpu')
                preds_test += [coords]
        preds_test = torch.cat(preds_test)
        preds_test = preds_test.data.numpy()
        df_test_result = df_test.copy()
        cols = ds_test.cols
        for idx, col in enumerate(cols):
            df_test_result[f'{col}_pred'] = preds_test[:,idx//2,idx%2]

        df_test_result.to_csv(output_dir/'result.csv',index=False)

        model_single.sample_size = nn.Parameter(torch.tensor(size),requires_grad=False)
        x_sample = torch.rand((2,3,size,size))
        if device_data:
            x_sample = x_sample.to(device_data)
        model_scripted = torch.jit.trace(model_single,x_sample)
        model_scripted.to('cpu')
        model_scripted.save(str(output_dir/model_output_name))


    def run_keypoint(self, image_dir, output_dir, model_path):
        
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True,exist_ok=True)

        if len(self._devices) >= 1:
            device = self._devices[0]
        else:
            device = 'cpu'
        model = torch.jit.load(str(Path(model_path).resolve()),map_location='cpu')
        model.to(device)
        size = model.sample_size.data.item()

        paths = list(image_dir.glob('**/*'))

        paths_valid = []
        for path in paths:
            p = image_verify(path)
            if p:
                paths_valid += [p]
        paths = paths_valid
        df = pd.DataFrame({'image':paths, 'label':[-1]*len(paths)})

        from dataset.keypoint_dataset import KeypointDataset
        ds = KeypointDataset(df,image_dir,height=size,width=size, p_aug=0)
        dataloader = DataLoader(ds, shuffle=False, 
                                batch_size=self._batch_size, num_workers=self._num_workers)

        model.eval()
        results = []
        for batch, _ in tqdm(dataloader):
            batch = batch.to(device)
            with torch.no_grad():
                coords, heatmaps = model(batch)
                coords = (coords+1)/2
                coords = coords.to('cpu')
                coords = coords.data.numpy()
                coords = coords.clip(0.01,0.99)
            for path, coord in zip(paths_valid,coords):
                output_path = output_dir/path
                output_path = output_path.with_suffix('.keypoint.csv')
                output_path.parent.mkdir(parents=True,exist_ok=True)
                result = {}
                for i, xy in enumerate(coord):
                    result[f'pred_{i:02d}_x'] = float(xy[0])
                    result[f'pred_{i:02d}_y'] = float(xy[1])
                results += [result]

                result_output = {
                    'x':[],
                    'y':[],
                }
                for i, xy in enumerate(coord):
                    result_output['x'] += [float(xy[0])]
                    result_output['y'] += [float(xy[1])]
                pd.DataFrame(result_output).to_csv(output_path)
            paths_valid = paths_valid[len(batch):]


if __name__ == '__main__':
    fire.Fire(VisionRunner)