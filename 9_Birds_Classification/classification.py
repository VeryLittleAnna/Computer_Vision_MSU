import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, random_split
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from torch import tensor
from PIL import Image
import cv2


import albumentations as A
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.augmentations.transforms import RandomBrightnessContrast
from albumentations.augmentations.transforms import RGBShift
from albumentations.augmentations.blur.transforms import GaussianBlur
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


BATCH_SIZE = 32
MAX_EPOCHES = 20
BASE_LR = 5e-5

IMG_SIZE = 256

DEFAULT_TRANSFORM = A.Compose(A.Normalize(mean=MEAN, std=STD))

TRANSFORMS = A.Compose([
        A.HorizontalFlip(p=0.5),
        Rotate(limit=25, p=0.7),
        RGBShift(r_shift_limit=15/255, g_shift_limit=15/255, b_shift_limit=15/255),
        RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        # GaussianBlur(blur_limit=3, p=0.3)
])

class MyDataset(Dataset):
    def __init__(self, data_gt, data_dir, train_size=0.9, transform=DEFAULT_TRANSFORM, mode='train', seed=42):
        self.transform = transform
        self.samples = []
        
        items_in_classes = {}
        for file_name, cl in data_gt.items():
            if cl in items_in_classes:
                items_in_classes[cl].append(file_name)
            else:
                items_in_classes[cl] = [file_name]
        self.paths = []
        self.classes = []
        for cl, filenames in items_in_classes.items():
            np.random.seed(seed)
            indexes = np.arange(len(filenames))
            np.random.shuffle(indexes)
            if mode == 'train':
                indexes = indexes[:int(train_size * len(filenames))]
            else:
                indexes = indexes[int(train_size * len(filenames)): ]
            for ind in indexes:
                img_path = os.path.join(data_dir, filenames[ind])
                self.paths.append(img_path)
                self.classes.append(cl)
        print(f"End of __init__: len = {len(self.paths)}")
            

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255
        image = self.transform(image=image)['image']
        image = torch.tensor(image.transpose(2, 0, 1))
        return image, self.classes[index]



class ResnetClassifier(pl.LightningModule):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()        
        self.resnet_model = resnet50(pretrained=pretrained)
        linear_size_in = list(self.resnet_model.children())[-1].in_features
        self.resnet_model.fc = nn.Linear(linear_size_in, num_classes)

        for child in list(self.resnet_model.children()):
                for param in child.parameters():
                    param.requires_grad = True

        for child in list(self.resnet_model.children())[:-4]:
            for param in child.parameters():
                param.requires_grad = False
 
    def forward(self, x):
        return F.log_softmax(self.resnet_model(x), dim=1)
    
class MyModel(pl.LightningModule):

    def __init__(self, lr_rate=BASE_LR, freeze='most', pretrained=False):
        super(MyModel, self).__init__()
        self.model = ResnetClassifier(50, pretrained=pretrained)

        self.lr_rate = lr_rate

    def forward(self, x):
      return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}

        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, prog_bar=True)

        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]

        self.log('val_loss', loss, on_step=True, on_epoch=False)
        self.log('val_acc', acc, on_step=True, on_epoch=False)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]

        self.log('test_loss', loss, on_step=True, on_epoch=False)
        self.log('test_acc', acc, on_step=True, on_epoch=False)

        return {'test_loss': loss, 'test_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        print('Accuracy: ', round(float(avg_acc), 3))
        self.log('val_acc', avg_acc, on_epoch=True, on_step=False)
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        self.log('test_acc', avg_acc, on_epoch=True, on_step=False)
        return {'test_loss': avg_loss, 'test_acc': avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        return [optimizer]



def train_classifier(train_gt, train_img_dir, fast_train=False, ignore=False):
    model = MyModel(pretrained=(not fast_train))
    if fast_train:
      AUG_CNT = 0
      BATCH_SIZE = 64
    if ignore:
      return model
    workers = 0 if fast_train else 12
    train_dataset = MyDataset(train_gt, train_img_dir, mode='train', transform=TRANSFORMS)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    if fast_train:
        trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
        trainer.fit(model, train_dataloader)
    else:
        MyModelCheckpoint = ModelCheckpoint(dirpath='.',
                                            filename='birds_model',
                                            monitor='val_acc', 
                                            mode='max', 
                                            save_top_k=1)
        MyEarlyStopping = EarlyStopping(monitor = "val_acc",
                                        mode = "max",
                                        patience = 2,
                                        verbose = True)

        valid_dataset = MyDataset(train_gt, train_img_dir, mode='val')
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE // 2, shuffle=False, num_workers=workers)
        
        trainer = pl.Trainer(accelerator='gpu', callbacks=[MyModelCheckpoint, MyEarlyStopping], devices=1, max_epochs=MAX_EPOCHES) #, default_root_dir="./")
        trainer.fit(model, train_dataloader, valid_dataloader)

    return model

def classify(model_filename, test_img_dir):
    model = MyModel.load_from_checkpoint(model_filename)
    model.eval()
    ans = {}
    for name in os.listdir(test_img_dir):
        image = np.array(Image.open(os.path.join(test_img_dir, name)).convert("RGB"), dtype=np.float32)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255
        image = DEFAULT_TRANSFORM(image=image)['image']
        image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0)
        predict = model(image)
        cl = predict.detach()[0].numpy()
        ans[name] = np.argmax(cl)
    return ans
