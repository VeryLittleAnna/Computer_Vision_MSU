# -*- coding: utf-8 -*-
import torch
import torchvision
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import csv
import json
import tqdm
import pickle
import typing
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsClassifier
from torch.nn import functional as F


CLASSES_CNT = 205

#здесь работа уже с "хорошим" преобработанным датасетом


class DatasetRTSD(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения датасета.
    :param root_folders: список путей до папок с данными
    :param path_to_classes_json: путь до classes.json
    """
    def __init__(self, root_folders, path_to_classes_json) -> None:
        super(DatasetRTSD, self).__init__()
        self.classes, self.class_to_idx = self.get_classes(path_to_classes_json)
        self.samples = [] ### YOUR CODE HERE - список пар (путь до картинки, индекс класса)
        self.classes_to_samples = {} ### YOUR CODE HERE - cловарь из списков картинок для каждого класса, classes_to_samples[индекс класса] = [список чисел-позиций картинок в self.samples]
        self.transform = [] ### YOUR CODE HERE - аугментации + нормализация + ToTensorV2
        self.classes_to_samples = {ind: [] for ind in self.class_to_idx.values()}
        for root_folder in root_folders:
            for class_folder in os.listdir(root_folder):
                cur_folder = os.path.join(root_folder, class_folder) 
                cur_ind = self.class_to_idx[class_folder]
                for image_name in os.listdir(cur_folder):
                    cur_path = os.path.join(cur_folder, image_name)
                    self.classes_to_samples[cur_ind].append(len(self.samples))
                    self.samples.append((cur_path, cur_ind))
        # self.transform = A.Compose([
        #     A.Rotate(limit=30, p=0.5),
        #     A.HorizontalFlip(p=0.5),
        #     #A.RandomBrightnessContrast(p=0.3)
        # ])
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ConvertImageDtype(torch.float)
            #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        path, cl = self.samples[index]
        image = torchvision.io.read_image(path)
        image = self.preprocess(image)
        # image = self.transform(image=image)['image']
        return image, path, cl

    def __len__(self):
        return len(self.samples)


    @staticmethod
    def get_classes(path_to_classes_json):
        """
        Считывает из classes.json информацию о классах.
        :param path_to_classes_json: путь до classes.json
        """
        class_to_idx = {} ### YOUR CODE HERE - словарь, class_to_idx['название класса'] = индекс
        classes = [] ### YOUR CODE HERE - массив, classes[индекс] = 'название класса'
        fin = open(path_to_classes_json)
        data = json.load(fin)
        for name, value in data.items():
            class_to_idx[name] = value['id']
        classes = [-1] * len(class_to_idx)
        for name, num in class_to_idx.items():
            classes[num] = name
        return classes, class_to_idx


class TestData(torch.utils.data.Dataset):
    """
    Класс для чтения и хранения тестового датасета.
    :param root: путь до папки с картинками знаков
    :param path_to_classes_json: путь до classes.json
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    def __init__(self, root, path_to_classes_json, annotations_file=None):
        super(TestData, self).__init__()
        self.root = root
        self.samples = [] ### YOUR CODE HERE - список путей до картинок
        self.transform = [] ### YOUR CODE HERE - преобразования: ресайз + нормализация + ToTensorV2
        self.targets = None
        if annotations_file is not None:
            self.targets = {} ### YOUR CODE HERE - словарь, targets[путь до картинки] = индекс класса
            with open(annotations_file) as fil:
                f_csv = csv.reader(fil, delimiter=',')
                next(f_csv)
                self.targets = dict(f_csv)

        self.samples = list(os.listdir(root))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ConvertImageDtype(torch.float)
            # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        """
        Возвращает тройку: тензор с картинкой, путь до файла, номер класса файла (если нет разметки, то "-1").
        """
        image_name = self.samples[index]
        image_path = os.path.join(self.root, image_name)
        image = torchvision.io.read_image(image_path) 
        image = self.transform(image)
        return image, image_name, (self.targets.get(image_name, -1) if self.targets is not None else -1)
    
    def __len__(self):
        return len(self.samples)

def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)

class CustomNetwork(pl.LightningModule):
    """
    Класс, реализующий нейросеть для классификации.
    :param features_criterion: loss-функция на признаки, извлекаемые нейросетью перед классификацией (None когда нет такого лосса)
    :param internal_features: внутреннее число признаков
    """
    def __init__(self, features_criterion = None, internal_features = 1024, pretrained=False):
        super(CustomNetwork, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        linear_size_out = list(self.model.children())[-1].in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features=linear_size_out, out_features=internal_features),
            nn.ReLU(),
            nn.Linear(in_features=internal_features, out_features=205)
        )

        for child in list(self.model.children()):
            for param in child.parameters():
                param.requires_grad = True
        ### YOUR CODE HERE

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, _, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}

        acc = torch.sum(logits.argmax(dim=1) == y) / y.shape[0]
        self.log('train_acc', acc, on_step=True, prog_bar=True)
        self.log('train_loss', loss, on_step=True, prog_bar=True)

        return {'loss': loss, 'log': logs}

    def predict(self, x):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param x: батч с картинками
        """
        pred = self.model(x)
        return torch.argmax(pred, dim=1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return [optimizer]


def train_simple_classifier():
    """Функция для обучения простого классификатора на исходных данных."""
    ### YOUR CODE HERE
    model = CustomNetwork(pretrained=False)
    dataset_train = DatasetRTSD(root_folders=['./output'], path_to_classes_json='./classes.json')
    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=100, num_workers=16)
    trainer = pl.Trainer(accelerator="auto",
                        devices=1 if torch.cuda.is_available() else None,
                        max_epochs=1) #, logger=False, checkpoint_callback=False)
    trainer.fit(model, dataloader_train)
    torch.save(model.state_dict(), "simple_model.pth")
    return model

# train_simple_classifier()

def apply_classifier(model, test_folder, path_to_classes_json):
    """
    Функция, которая применяет модель и получает её предсказания.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param path_to_classes_json: путь до файла с информацией о классах classes.json
    """
    results = [] ### YOUR CODE HERE - список словарей вида {'filename': 'имя файла', 'class': 'строка-название класса'}
    dataset_test = TestData(test_folder, path_to_classes_json=path_to_classes_json)
    # dataloader_test = DataLoader(dataloader_test, batch_size=32, shuffle=False)
    classes, _ = DatasetRTSD.get_classes(path_to_classes_json)
    for i, name in enumerate(dataset_test.samples):
        pred = model.predict(dataset_test[i][0][None, ...])
        results.append({'filename': name, 'class': classes[pred]}) #.detach()[0].numpy())
    return results


def test_classifier(model, test_folder, path_to_classes_json, annotations_file):
    """
    Функция для тестирования качества модели.
    Возвращает точность на всех знаках, Recall на редких знаках и Recall на частых знаках.
    :param model: модель, которую нужно протестировать
    :param test_folder: путь до папки с тестовыми данными
    :param annotations_file: путь до .csv-файла с аннотациями (опциональный)
    """
    ### YOUR CODE HERE
    # return total_acc, rare_recall, freq_recall
    pass



class SignGenerator(object):
    """
    Класс для генерации синтетических данных.
    :param background_path: путь до папки с изображениями фона
    """
    def __init__(self, background_path):
        ### YOUR CODE HERE
        pass

    def get_sample(self, icon):
        """
        Функция, встраивающая иконку на случайное изображение фона.
        :param icon: Массив с изображением иконки
        """
        icon = ... ### YOUR CODE HERE
        bg = ... ### YOUR CODE HERE - случайное изображение фона
        return ### YOUR CODE HERE


def generate_one_icon(args):
    """
    Функция, генерирующая синтетические данные для одного класса.
    :param args: Это список параметров: [путь до файла с иконкой, путь до выходной папки, путь до папки с фонами, число примеров каждого класса]
    """
    ### YOUR CODE HERE


def generate_all_data(output_folder, icons_path, background_path, samples_per_class = 1000):
    """
    Функция, генерирующая синтетические данные.
    Эта функция запускает пул параллельно работающих процессов, каждый из которых будет генерировать иконку своего типа.
    Это необходимо, так как процесс генерации очень долгий.
    Каждый процесс работает в функции generate_one_icon.
    :param output_folder: Путь до выходной директории
    :param icons_path: Путь до директории с иконками
    :param background_path: Путь до директории с картинками фона
    :param samples_per_class: Количество примеров каждого класса, которые надо сгенерировать
    """
    with ProcessPoolExecutor(8) as executor:
        params = [[os.path.join(icons_path, icon_file), output_folder, background_path, samples_per_class]
                  for icon_file in os.listdir(icons_path)]
        list(tqdm.tqdm(executor.map(generate_one_icon, params)))


def train_synt_classifier():
    """Функция для обучения простого классификатора на смеси исходных и ситетических данных."""
    ### YOUR CODE HERE
    return model


class FeaturesLoss(torch.nn.Module):
    """
    Класс для вычисления loss-функции на признаки предпоследнего слоя нейросети.
    """
    def __init__(self, margin: float) -> None:
        super(FeaturesLoss, self).__init__()
        ### YOUR CODE HERE
        pass


class CustomBatchSampler(torch.utils.data.sampler.Sampler[typing.List[int]]):
    """
    Класс для семплирования батчей с контролируемым числом классов и примеров каждого класса.
    :param data_source: Это датасет RTSD
    :param elems_per_class: Число элементов каждого класса
    :param classes_per_batch: Количество различных классов в одном батче
    """
    def __init__(self, data_source, elems_per_class, classes_per_batch):
        ### YOUR CODE HERE
        pass
    def __iter__(self):
        ### YOUR CODE HERE
        pass


def train_better_model():
    """Функция для обучения классификатора на смеси исходных и ситетических данных с новым лоссом на признаки."""
    ### YOUR CODE HERE
    return model


class ModelWithHead:
    """
    Класс, реализующий модель с головой из kNN.
    :param n_neighbors: Количество соседей в методе ближайших соседей
    """
    def __init__(self, n_neighbors):
        self.model = CustomNetwork(pretrained=False)
        ### YOUR CODE HERE


    def load_nn(self, nn_weights_path):
        """
        Функция, загружающая веса обученной нейросети.
        :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
        """
        ### YOUR CODE HERE
        self.model.load_state_dict(torch.load(nn_weights_path, map_location='cpu'))
        self.model.cpu().eval()
        
    def predict(self, imgs):
        """
        Функция для предсказания классов-ответов. Возвращает np-массив с индексами классов.
        :param imgs: батч с картинками
        """
        # features, model_pred = ... ### YOUR CODE HERE - предсказание нейросетевой модели
        # features = features / np.linalg.norm(features, axis=1)[:, None]
        # knn_pred = ... ### YOUR CODE HERE - предсказание kNN на features
        # return knn_pred
        return self.model.predict(imgs)

    def load_head(self, knn_path):
        """
        Функция, загружающая веса kNN (с помощью pickle).
        :param knn_path: Путь, откуда надо прочитать веса kNN
        """
        ### YOUR CODE HERE
        pass



class IndexSampler(torch.utils.data.sampler.Sampler[int]):
    """
    Класс для семплирования батчей с картинками индекса.
    :param data_source: Это датасет RTSD с синтетическими примерами
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    def __init__(self, data_source, examples_per_class) -> None:
        ### YOUR CODE HERE
        pass
    def __iter__(self):
        """Функция, которая будет генерировать список индексов элементов в батче."""
        return ### YOUR CODE HERE


def train_head(nn_weights_path, examples_per_class = 20):
    """
    Функция для обучения kNN-головы классификатора.
    :param nn_weights_path: Это путь до весов обученной нейросети с улучшенными признаками на предпоследнем слое
    :param examples_per_class: Число элементов каждого класса, которые должны попасть в индекс
    """
    ### YOUR CODE HERE
