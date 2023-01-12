
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

import os
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from PIL import Image
import torchvision.models
import albumentations as A

import torch.nn as nn
import torchvision.models


NUM_WORKERS = 16
MAX_EPOCHS = 10
IMG_SIZE = 192
BATCH_SIZE = 4

NORMALIZE = A.Normalize()
 #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

TRANSFORMS = A.Compose([
    A.Normalize(),
    A.Rotate(limit = 30, p = 1.0), #limit=30, p=1.0
    # A.HorizontalFlip(p = 0.5),
    # A.RGBShift(r_shift_limit=10, b_shift_limit=10, g_shift_limit=10, p=0.7),
    # A.RandomBrightness(p=0.7),
])

NUM_CLASSES = 200

class MyDataset(Dataset):
    def __init__(self, data_path, mode, transform=A.Normalize()):
        super(MyDataset, self).__init__()

        self.samples = []
        self.transform = transform
        self.data_dir = data_path
        self.mode = mode

        dir_names = sorted(os.listdir(os.path.join(data_path, "images")))
        for subdir_name in dir_names:
            images = sorted(os.listdir(os.path.join(data_path, "images", subdir_name)))
            if mode == 'train':
                images = images[:-1]
            elif mode == "valid":
                images = images[-1:]
            for filename in images:
                self.samples.append((
                    os.path.join(data_path, "images", subdir_name, filename), #image
                    os.path.join(data_path, "gt", subdir_name, filename[:-4]+".png") #gt - mask
                ))
    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = np.array(Image.open(image_path).convert('RGB'), dtype = np.float32)
        mask = np.array(Image.open(mask_path).convert('L'), dtype = np.float32)
        mask  = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255

        tranform_result = self.transform(image=image, mask=mask)
        image = torch.tensor(tranform_result['image'].transpose(2, 0, 1))
        mask = torch.tensor(tranform_result['mask']).unsqueeze(0)
        return image, mask

    def __len__(self):
        return len(self.samples)



def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = torchvision.models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
    
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +\
                                                 target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

class MyModel(pl.LightningModule):
    # REQUIRED
    def __init__(self, num_classes):
        super().__init__()
        """ Define computations here. """
        
        self.model = ResNetUNet(num_classes)
        
        # freeze backbone layers
        for l in self.model.base_layers:
            for param in l.parameters():
                param.requires_grad = False
        
        
        self.bce_weight = 0.9
    
    # REQUIRED
    def forward(self, x):
        """ Use for inference only (separate from training_step). """
        x = self.model(x)
        return x
    
    
    # REQUIRED
    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch

        y_logit = self(x)        
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)

        return {'loss': loss}
    
    # REQUIRED
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=5e-4)
        # optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  mode='min', 
                                                                  factor=0.1, 
                                                                  patience=1, 
                                                                  verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        } 
        
        return [optimizer], [lr_dict]
    
    # OPTIONAL
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        x, y = batch

        y_logit = self(x)        
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)
        
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)

        return {'val_loss': loss, 'logs':{'dice':dice, 'bce': bce}}

    # OPTIONAL
    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        print(f"| Train_loss: {avg_loss:.3f}" )
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
     
    # OPTIONAL
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        avg_dice = torch.stack([x['logs']['dice'] for x in outputs]).mean()
        avg_bce = torch.stack([x['logs']['bce'] for x in outputs]).mean()
        
        print(f"[Epoch {self.trainer.current_epoch:3}] Val_loss: {avg_loss:.3f}, Val_dice: {avg_dice:.3f}, Val_bce: {avg_bce:.3f}", end= " ")
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)


def train_segmentation_model(train_data_path):
    model = MyModel(num_classes=2)
    train_dataset = MyDataset(train_data_path, mode='train', transform=TRANSFORMS)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    val_dataset = MyDataset(train_data_path, mode='valid')
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks import ModelCheckpoint

    ## Save the model periodically by monitoring a quantity.
    MyModelCheckpoint = ModelCheckpoint(dirpath='.',
                                        filename='segmentation_model.pth',
                                        monitor='val_loss', 
                                        mode='min', 
                                        save_top_k=1)

    ## Monitor a metric and stop training when it stops improving.
    MyEarlyStopping = EarlyStopping(monitor = "val_loss",
                                    mode = "min",
                                    patience = 2,
                                    verbose = True)
    trainer = pl.Trainer(
            accelerator = 'gpu',
            max_epochs = MAX_EPOCHS,
            # gpus = 1, 
            callbacks=[MyEarlyStopping, MyModelCheckpoint]
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), 'segmentation_model.pth')

def predict(model, img_path):
    model.eval()
    image = np.array(Image.open(img_path).convert('RGB'), dtype = np.float32)
    old_shape = image.shape[:2][::-1]
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255
    image = NORMALIZE(image=image)['image']
    image_tensor = torch.tensor(image.transpose(2, 0, 1))
    # image = torch.tensor(NORMALIZE(image_tensor)).unsqueeze(0)
    result = torch.sigmoid(model(image_tensor.unsqueeze(0)))
    result = cv2.resize(result.squeeze().detach().numpy(), old_shape)
    return result

def get_model():
    return MyModel(num_classes=2)