from __future__ import annotations

import torch
import torch_directml
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, transforms

import matplotlib.pyplot as plt
from pathlib import Path


def select_device(device_name=''):
    if device_name.lower() == 'cuda':
        if torch.cuda.is_available():
            print("Using CUDA")
            return torch.device('cuda:0')
        else:
            print('CUDA is not available')
    if device_name.lower() == 'dml':
        if torch_directml.is_available():
            print("Using DirectML")
            return torch_directml.device(torch_directml.default_device())
        else:
            print('DirectML is not available')
    print('Falling back to CPU')
    return torch.device('cpu')


def display_sample_images(dataset: list | Dataset, sample_rows=3, sample_cols=3):
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, sample_rows * sample_cols + 1):
        fig.add_subplot(sample_rows, sample_cols, i)
        img, label = dataset[torch.randint(len(dataset), size=(1,)).item()]
        plt.axis('off')
        plt.title(label)
        plt.imshow(img.permute(1, 2, 0))
    plt.show()


class CNN(nn.Module):
    def __init__(self, input_size=(3, 224, 224), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_channels = 1 if len(input_size) == 2 else input_size[0]
        self.input_dim = input_size if len(input_size) == 2 else input_size[1:]
        self.fully_connected_features = 128

        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, kernel_size=3, out_channels=32, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, kernel_size=3, out_channels=32, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, kernel_size=3, out_channels=32, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()

        shadow_input = torch.randn(input_size)
        fc_input_features = self.flatten(self.conv_pool(shadow_input)).size(-1)

        self.fully_connected = nn.Linear(fc_input_features, self.fully_connected_features)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.fully_connected_features, 2),
            nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_pool(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        return self.out(x)


if __name__ == '__main__':
    data_dir = Path("../data/udemy_dataset_1")
    train_data_dir = data_dir / 'training_set'
    test_data_dir = data_dir / 'test_set'
    BATCH_SIZE = 32
    RESIZE_WIDTH = 256
    RESIZE_HEIGHT = 256
    AFFINE_TRANSFORM = {
        'DEGREE': (-15, 15),
        'SHEAR': (0.3, 0.3)
    }
    NETWORK_BASE_WIDTH = 224
    NETWORK_BASE_HEIGHT = 224

    train_transform = Compose([
        transforms.Resize((RESIZE_WIDTH, RESIZE_HEIGHT)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(
            degrees=AFFINE_TRANSFORM['DEGREE'],
            shear=AFFINE_TRANSFORM['SHEAR']
        ),
        transforms.RandomCrop((NETWORK_BASE_WIDTH, NETWORK_BASE_HEIGHT)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x * 1.0 / 255),
    ])

    test_transform = Compose([
        transforms.Resize((NETWORK_BASE_WIDTH, NETWORK_BASE_HEIGHT)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x * 1.0 / 255),
    ])

    train_dataset = ImageFolder(root=str(train_data_dir), transform=train_transform)
    test_dataset = ImageFolder(root=str(test_data_dir), transform=test_transform)

    # display_sample_images(train_dataset, 5, 5)
    # display_sample_images(test_dataset)
    # print(train_dataset[0][0].size())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    net = CNN()
    print(net)
