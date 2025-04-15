import torch
from torch import nn
import pandas as pd
from datasets import load_dataset
import numpy as np

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
    def train_model_mannually(cls):
        df = cls.train_data()
        input_features = df.drop("actual", axis=1).values
        labels = df["actual"].values

        model = WeatherModel(input_features, labels)

        input_size = input_features.shape[1]
        hidden_size = 128
        output_size = 1
        learning_rate = 0.001

        weights = torch.randn((input_size, hidden_size), dtype=float, required_grad=True)
        biases = torch.randn((1, hidden_size), dtype=float, required_grad=True)

        weights2 = torch.randn((hidden_size, 1), dtype=float, required_grad=True)
        biases2 = torch.randn((output_size, 1), dtype=float, required_grad=True)
        losses = []

        for i in range(1000):
            # 计算隐层，加入激活函数
            hidden = model.x.mm(weights) + biases
            hidden = torch.relu(hidden)
            # 预测结果
            predictions = hidden.mm(weights2) + biases2
            # 计算损失
            loss = torch.mean((predictions - model.y) ** 2)
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

    @classmethod
    def train_model_simple(cls):
        df = cls.train_data()
        input_features = df.drop("actual", axis=1).values
        labels = df["actual"].values

        input_size = input_features.shape[1]
        hidden_size = 128
        output_size = 1
        batch_size = 16
        learning_rate = 0.001

        my_nn = nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_size, output_size),
        )

        cost = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(my_nn.parameters(), lr=learning_rate)

        losses = []
        for i in range(1000):
            batch_loss = []
            for start in range(0, len(input_features), batch_size):
                end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
                xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
                yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
                prediction = my_nn(xx)
                loss = cost(prediction, yy)
                batch_loss.append(loss.data.numpy())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            if i % 100 == 0:
                mean_batch_loss = np.mean(batch_loss)
                losses.append(mean_batch_loss)
                print(f"Epoch {i}: Loss {mean_batch_loss}")
