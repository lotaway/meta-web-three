from typing import Optional

import torch
from torch import nn
from utils import get_device
from torchvision import datasets, transforms

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        """
        in_features : 真实数据的维度、同时也是生成的假数据的
        """
        super().__init__()
        self.disc = nn.Sequential(nn.Linear(in_features, out_features),
                                  nn.LeakyReLU(0.1),  # 由于生成对抗网络的损失非常容易梯度消失，因此使用LeakyReLU
                                  nn.Linear(128, 1),
                                  nn.Sigmoid()
                                  )

    def forward(self, data):
        "输入的data可以是真实数据时，Disc输出dx。输入的data是gz时，Disc输出dgz"
        return self.disc(data)

    def load_model(self, path: str | None = "model.pkl"):
        self.load_state_dict(torch.load(path))

    @classmethod
    def train_data(cls):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        x_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0  # 归一化并展平
        y_train = train_dataset.targets.numpy()  # 标签
        return x_train, y_train

    @classmethod
    def train_model(cls, x_train, y_train):
        # input_dim = 1
        # output_dim = 1
        device = get_device()
        # self.model = torch.nn.Linear(10, 5).to(device)
        in_features = 784
        out_features = 128
        epochs = 1000
        learn_rate = 0.01
        model = LinearRegressionModel(in_features, out_features).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)
        criterion = nn.MSELoss()
        for epoch in range(epochs.__or__(epochs)):
            epoch += 1
            inputs = torch.from_numpy(x_train).to(device)
            labels = torch.from_numpy(y_train).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"epoch: {epoch}, loss: {loss.item()}")

    def save_model(self, path: str | None = "model.pkl"):
        torch.save(self.disc.state_dict(), path)
