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
    def __init__(self, input_size=(3, 64, 64), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = Path('../model/CNN.pth')
        self.input_channels = 1 if len(input_size) == 2 else input_size[0]
        self.input_dim = input_size if len(input_size) == 2 else input_size[1:]
        self.fully_connected_features = 256

        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, kernel_size=9, out_channels=32, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, kernel_size=9, out_channels=32, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()

        shadow_input = torch.randn([1, *input_size])
        fc_input_features = self.flatten(self.conv_pool(shadow_input)).size(-1)

        self.fully_connected = nn.Linear(fc_input_features, self.fully_connected_features)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.fully_connected_features, 1),
            # nn.ReLU(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_pool(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        x = self.dropout(x)
        return self.out(x)


if __name__ == '__main__':
    data_dir = Path("../data/udemy_dataset_1")
    train_data_dir = data_dir / 'training_set'
    test_data_dir = data_dir / 'test_set'
    BATCH_SIZE = 64
    RESIZE_WIDTH = 70
    RESIZE_HEIGHT = 70
    AFFINE_TRANSFORM = {
        'DEGREE': (-15, 15),
        'SHEAR': (0.3, 0.3)
    }
    NETWORK_BASE_WIDTH = 64
    NETWORK_BASE_HEIGHT = 64

    train_transform = Compose([
        transforms.Resize((NETWORK_BASE_WIDTH, NETWORK_BASE_HEIGHT)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x * 1.0 / 255),
    ])

    test_transform = Compose([
        transforms.Resize((NETWORK_BASE_WIDTH, NETWORK_BASE_HEIGHT)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x * 1.0 / 255),
    ])

    train_dataset = ImageFolder(root=str(train_data_dir), transform=train_transform)
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])
    test_dataset = ImageFolder(root=str(test_data_dir), transform=test_transform)

    # display_sample_images(train_dataset, 5, 5)
    # display_sample_images(test_dataset)
    # print(train_dataset[0][0].size())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    net = CNN()
    print(net)

    EPOCHS = 50
    DEVICE = select_device("dml")

    net = net.to(DEVICE)
    loss_fn = nn.BCELoss()
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    TRAIN = (1 == 1)

    if TRAIN:
        for epoch in range(1, EPOCHS + 1):
            epoch_loss = 0
            for batch_no, (images, labels) in enumerate(train_loader):
                net.train()
                images = images.to(DEVICE)
                labels = labels.to(DEVICE, dtype=torch.float)

                predictions = net(images)
                loss = loss_fn(predictions.squeeze(), labels)

                loss.backward()
                opt.step()
                opt.zero_grad()

                epoch_loss += loss.item()
                if batch_no % 100 == 0:
                    # CALCULATE ACCURACY USING VALIDATION DATASET #
                    total_images = len(validation_dataset)
                    accurate = 0
                    net.eval()
                    for v_img, v_label in validation_loader:
                        v_img, v_label = v_img.to(DEVICE), v_label.to(DEVICE)
                        validation_predictions = (net(v_img).squeeze() > 0.5).float()
                        acc_matrix = torch.Tensor(validation_predictions == v_label)
                        accurate += acc_matrix.sum().item()
                    accuracy = 100 * accurate / total_images
                    print(f"[e]: {epoch:4d}, [b]: {batch_no:4d}, [l]: {loss.item():>2.4f}, [a]: {accuracy:>2.4f}")

            print(f"[e]: {epoch:4d}, [l]: {epoch_loss:>2.4f}")
