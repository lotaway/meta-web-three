import torch
from torch import nn
import pandas as pd
from datasets import load_dataset

class WeatherModel(nn.Module):
    def __init__(self, input_features, labels):
        super.__init__()

        self.x = torch.tensor(input_features, dtype=float)
        self.y = torch.tensor(labels, dtype=float)

    @classmethod
    def train_data(cls):
        dataset = load_dataset("kanishka089/weather")
        df = pd.read_csv(dataset)
        return df

    @classmethod
    def train_model(cls):
        df = cls.train_data()
        input_features = df.drop("actual", axis=1).values
        labels = df["actual"].values

        model = WeatherModel(input_features, labels)

        weights = torch.randn((14, 128), dtype=float, required_grad=True)
        biases = torch.randn((1, 128), dtype=float, required_grad=True)

        weights2 = torch.randn((128, 1), dtype=float, required_grad=True)
        biases2 = torch.randn((1, 1), dtype=float, required_grad=True)

        learning_rate = 0.001
        losses = []

        for i in range(1000):
            # 计算隐层，加入激活函数
            hidden = model.x.mm(weights) + biases
            hidden = torch.relu(hidden)
            # 预测结果
            predictions = hidden.mm(weights2) + biases2
            # 计算损失
            loss = torch.mean((predictions - y) ** 2)
            losses.append(loss.data.numpy())

            if i % 100 == 0:
                print(f"Epoch {i}: Loss {loss.data.numpy()}")

            # 更新参数
            weights.data.add_(-learning_rate * weights.grad.data)
            biases.data.add_(-learning_rate * biases.grad.data)
            weights2.data.add_(-learning_rate * weights2.grad.data)
            biases2.data.add_(-learning_rate * biases2.grad.data)

            # 清空梯度
            weights.grad.data.zero_()
            biases.grad.data.zero_()
            weights2.grad.data.zero_()
            biases2.grad.data.zero_()