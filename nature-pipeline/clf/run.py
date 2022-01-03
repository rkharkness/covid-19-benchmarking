#!/usr/bin/env python3
import fire
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fastai.metrics import add_metrics
from fastai.callback import Callback

import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score


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


class AUCk(Callback):
    def __init__(self, class_y):
        self.name = f'AUC({class_y})'
        self._class_y = class_y


    def on_epoch_begin(self, **kwargs):
        self._label = []
        self._score = []
        

    def on_batch_end(self, last_output, last_target, **kwargs):
        self._label += [(last_target == self._class_y).to('cpu').data.numpy()]
        self._score += [last_output[:,self._class_y].to('cpu').data.numpy()]    


    def on_epoch_end(self, last_metrics, **kwargs):
        label = np.concatenate(self._label)
        score = np.concatenate(self._score)
        if len(set(label)) != 2:
            auc = 0.5
        else:
            auc = roc_auc_score(label,score)
        return add_metrics(last_metrics, auc)


def model_func_with_pretrained(*args, **kargs):
    model_func = kargs.pop('model_func')
    pretrained_path = kargs.pop('pretrained_path')
    model = model_func(*args, **kargs)
    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


class VisionRunner(object):
    def __init__(self, lr=1e-3, valid_ratio=0.2, n_epoch=10, 
    #                 # patience=None,
                    size=224, batch_size=32, num_workers=0, device=None,
                    monitor='valid_loss', unfreeze=False, 
                    pretrained=None, model='models.resnet50'):
        super(VisionRunner, self).__init__()
        self._devices = parse_device(device)
        self._size = size
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._n_epoch = n_epoch
        self._monitor = monitor
        self._lr = lr
        self._valid_ratio = valid_ratio

        self._model = model
        self._unfreeze = unfreeze
        self._pretrained = pretrained


    def train(self, df_path, data_root, output_dir, col_image='image', col_label='label'):

        import matplotlib
        matplotlib.use('Agg')
        from fastai.vision import Learner
        from fastai.vision import cnn_learner
        from fastai.vision import get_transforms, models
        from fastai.vision import accuracy, AUROC
        from fastai.vision import DataBunch, ImageList, DatasetType
        from fastai.callbacks import SaveModelCallback
        from fastai.callbacks import EarlyStoppingCallback
        
        data_root = Path(data_root)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)

        df = pd.read_csv(df_path)
        df_train = df.query('dataset == "train"')
        df_valid = df.query('dataset == "valid"')
        df_test = df[~df.dataset.isin(['train','valid'])]

        size = self._size
        num_workers = self._num_workers
        batch_size = self._batch_size
        n_epoch = self._n_epoch
        devices = self._devices
        pretrained = self._pretrained

        print(devices)
        if len(devices) != 0 and devices[0].lower() != 'cpu':
            pin_memory = True
            device_data = devices[0]
        else:
            pin_memory = False
            device_data = None
        
        idx_train = df_train.index
        idx_valid = df_valid.index
        idx_test = df_test.index

        model_backbone = eval(self._model)

        if not (pretrained is None):
            model_backbone = partial(model_func_with_pretrained,
                                    model_func=model_backbone,pretrained_path=pretrained)

        num_classes = df[col_label].nunique()
        data = (ImageList.from_df(df=df,folder='.',path=data_root,cols=col_image)
                .split_by_idxs(idx_train,idx_valid)
                .label_from_df(col_label)
                .transform(get_transforms(), size=(size,size))
                .databunch(bs=batch_size, device=device_data)
               )
        data_test = (ImageList.from_df(df=df,folder='.',path=data_root,cols=col_image)
                .split_by_idxs(idx_test,idx_test)
                .label_from_df(col_label)
                .transform(get_transforms(), size=(size,size))
                .databunch(bs=batch_size, device=device_data)
               )
        data.add_test(data_test.valid_ds.x)
        loss_func = nn.CrossEntropyLoss()
        loss_func = loss_func.to(devices[0])

        metrics = [accuracy]
        metrics += [AUCk(i) for i in range(num_classes)]
        learn = cnn_learner(data, model_backbone, metrics=metrics, path=output_dir, loss_func=loss_func)

        if self._unfreeze:
            learn.unfreeze()

        model_single = learn.model
        if len(devices) >= 2:
            learn.model = nn.DataParallel(model_single,device_ids=devices)

        lr = self._lr
        monitor = self._monitor
        callbacks = [SaveModelCallback(learn, every='improvement',monitor=monitor, name='best')]
        learn.fit_one_cycle(n_epoch, slice(lr), pct_start=0.3, callbacks=callbacks)

        x_sample = torch.rand((2,3,size,size))
        x_sample = x_sample.to(devices[0])

        model_single.sample_size = nn.Parameter(torch.tensor(self._size),requires_grad=False)
        model_scripted = torch.jit.trace(model_single,x_sample)
        model_scripted.to('cpu')
        model_scripted.save(str(output_dir/'scripted_model.zip'))


    def infer(self, data_dir, output_path, model_path):
        data_dir = Path(data_dir)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True,exist_ok=True)
        if len(self._devices) > 0:
            device = self._devices[0]
        else:
            device = 'cpu'
        model = torch.jit.load(str(Path(model_path).resolve()),map_location='cpu')
        model.to(device)

        paths = list(data_dir.glob('**/*'))
        paths = [p.resolve() for p in paths]
        paths_valid = []
        for path in paths:
            p = image_verify(path)
            if p:
                paths_valid += [p]
        paths = paths_valid
        df = pd.DataFrame({'image':paths, 'label':[-1]*len(paths)})

        from imagedataset import ImageDataset
        if hasattr(model,'sample_size'):
            size = model.sample_size.data.item()
        else:
            size = self._size
        dataset = ImageDataset(paths,data_dir,size=size)
        dataloader = DataLoader(dataset, shuffle=False, 
                                batch_size=self._batch_size, num_workers=self._num_workers)

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                preds = model(batch_x)
                preds = nn.functional.softmax(preds,dim=1)
                preds = preds.to('cpu').data.numpy()
                for path, pred in zip(paths,preds):
                    result = {}
                    result['path'] = path
                    result['pred'] = int(pred.argmax(dim=1))
                    print(result)


if __name__ == '__main__':
    fire.Fire(VisionRunner)