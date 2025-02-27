import torch
import torch_directml
import matplotlib.pyplot
# from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import math
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn


def get_device():
    device = torch_directml.device()
    if device:
        print("Using GPU ms ml")
        return device
    # check if GPU device work
    match True:
        case torch.backends.mps.is_available():  # apple silicon
            print("Using GPU mps")
            return torch.device("mps")  # use AMD Metal Performance Shaders ?
        case torch.cuda.is_available():  # nvidia
            print("Using GPU nvidia")
            return torch.device("cuda:0")
        case torch.hip.is_available():  # amd
            print("Using GPU hip")
            return torch.device("hip")
        case _:
            print("Using CPU")
            return torch.device("cpu")

def load_data():
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载训练和测试数据集
    train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='mnist_data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def start_train():
    # 实例化模型、定义优化器和损失函数
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # torch.nn.Module
    model = SimpleNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(5):
        train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch [{epoch + 1}/5] completed.')

def paint():
    # 获得0到2π之间的ndarray对象
    x = np.arange(0, math.pi * 2, 0.05)
    y = np.sin(x)
    plt.plot(x, y)
    plt.xlabel("angle")
    plt.ylabel("sine")
    plt.title('sine wave')
    # 使用show展示图像
    plt.show()

def paint2():
    # Generate 100 random data points along 3 dimensions
    x, y, scale = np.random.randn(3, 100)
    fig, ax = plt.subplots()

    # Map each onto a scatterplot we'll create with Matplotlib
    ax.scatter(x=x, y=y, c=scale, s=np.abs(scale) * 500)
    ax.set(title="Some random data, created with JupyterLab!")
    plt.show()


def main():
    device = get_device()
    tensor = torch.randn(2, 2).to(device)
    print(tensor)
    paint()
    paint2()


if __name__ == "__main__":
    main()
