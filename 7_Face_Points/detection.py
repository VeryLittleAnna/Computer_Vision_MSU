import numpy as np
import pytorch_lightning as pl
import torch
from torch import tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision 
from torch.utils.data import Dataset
import os
from PIL import Image
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import random

IMG_SIZE=64


class MyDataset(Dataset):
    def __init__(self, train_data, data_dir, train_size=0.9, mode='train'):
        self.len = len(train_data)
        self.data = torch.zeros((self.len, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float)
        self.coords = torch.zeros((self.len, 28), dtype=torch.float)
        indices = np.arange(len(train_data))
        np.random.seed(42)
        np.random.shuffle(indices)
        for i, item in enumerate(train_data.items()):
            name, coords = item
            img = Image.open(os.path.join(data_dir, name)).convert("RGB")
            image_size = np.array(img).shape
            img = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype("float32")
            self.coords[i] = tensor(coords)
            self.coords[i, ::2] *= IMG_SIZE / image_size[1]
            self.coords[i, 1::2] *= IMG_SIZE / image_size[0]
            self.data[i] = tensor(np.transpose(img.astype("float32"), axes=(2, 0, 1)))
        self.data = self.data[indices]
        self.coords = self.coords[indices]
        if mode == 'train':
            self.data = self.data[:int(train_size * self.len), ...]
        else:
            self.data = self.data[int(train_size * self.len):]
        self.len = len(self.data)
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index], self.coords[index]

from torch import nn

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential( 
            nn.Conv2d(3, 64, 5, padding=2), 
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            # nn.Dropout(p=0.25),
            nn.MaxPool2d(2)
            )

        self.layer2 = nn.Sequential( 
            nn.Conv2d(64, 128, 5, padding=2), 
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            # nn.Dropout(p=0.25),
            nn.MaxPool2d(2)
            )
        self.layer3 = nn.Sequential( 
            nn.Conv2d(128, 256, 5, padding=2), 
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            # nn.Dropout(p=0.25),
            nn.MaxPool2d(2)
            )
        
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            # nn.Dropout(p=0.25),
            # nn.MaxPool2d(2)
            )
        self.layer5 = nn.Sequential(
            nn.Linear(256 * (IMG_SIZE // 8) ** 2, 64),
            nn.ReLU(),
            nn.Linear(64, 28)
        ) 

        self.loss = F.mse_loss

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, 1)     
        x = self.layer5(x)
        return x
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss(self(x), y)
        return loss
        # return {'loss': loss, 'acc': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss(self(x), y)
        return {'val_loss': loss}
        # return {'val_loss': loss, 'val_acc': loss} 

    # def configure_optimizers(self):
    #     return torch.optim.SGD(self.parameters(), lr=0.0001)
    
    # REQUIRED
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  mode='min', 
                                                                  factor=0.5, 
                                                                  patience=2, 
                                                                  verbose=True)
        lr_dict = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss"
        } 
        return [optimizer], [lr_dict]
    
    def training_epoch_end(self, outputs):
        # display average loss across epoch
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # print(f"Epoch {self.trainer.current_epoch}, Train_loss: {round(float(avg_loss), 3)}")
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
    
    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        # print(f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.2f}", end= " ")
        
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
    
    

def train_detector(train_data, train_img_dir, fast_train=True, ignore=False):
    model = MyModel()
    if ignore:
        return model
    train_dataset = MyDataset(train_data, train_img_dir, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=200)
    valid_dataset = MyDataset(train_data, train_img_dir, mode='val')
    valid_dataloader = DataLoader(valid_dataset, batch_size=100, shuffle=False, num_workers=1)
    if fast_train:
        trainer = pl.Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    else:
        MyModelCheckpoint = ModelCheckpoint(dirpath='.',
                                            filename='facepoints_model',
                                            monitor='train_loss', 
                                            mode='min', 
                                            save_top_k=10)
        MyEarlyStopping = EarlyStopping(monitor = "val_loss",
                                        mode = "min",
                                        patience = 2,
                                        verbose = True)

        trainer = pl.Trainer(accelerator="gpu", callbacks=[MyModelCheckpoint], devices=1, max_epochs=10, default_root_dir="./")
    trainer.fit(model, train_dataloader, valid_dataloader)

    return model

def detect(model_filename, test_img_dir):
    model = MyModel.load_from_checkpoint(model_filename)
    model.eval()
    ans = {}
    for name in os.listdir(test_img_dir):
        img = Image.open(os.path.join(test_img_dir, name)).convert("RGB")
        image_size = np.array(img).shape
        img = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype("float32")
        predict = model(torch.from_numpy(np.transpose(img, axes=(2, 0, 1))[None, ...]))
        coords = predict.detach()[0].numpy()
        coords[::2] *= image_size[1] / IMG_SIZE
        coords[1::2] *= image_size[0] / IMG_SIZE
        ans[name] = coords
    return ans
