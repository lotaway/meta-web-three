import matplotlib.pyplot
# from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import math

from utils import tokenize, get_device

def paint():
    # 获得0到2π之间的ndarray对象
    x = np.arange(0, math.pi * 2, 0.05)
    y = np.sin(x)
    plt.plot(x, y)
    plt.xlabel("angle")
    plt.ylabel("sine")
    plt.title('sine wave')
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
    tokens = tokenize("RagFlow对话系统特点与应用")
    print(tokens)


if __name__ == "__main__":
    main()
