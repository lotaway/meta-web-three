import torch
import torch_directml
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn

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
    return (train_loader, test_loader)

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
    (train_loader, test_loader) = load_data()

    # 开始训练
    for epoch in range(5):
        train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch [{epoch + 1}/5] completed.')