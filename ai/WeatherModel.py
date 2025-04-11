import torch
from torch import nn
import pandas as pd
from datasets import load_dataset

class WeatherModel(nn.Module):
    def __init__(self):
        super.__init__()

    @classmethod
    def train_data(cls):
        dataset = load_dataset("kanishka089/weather")
        df = pd.read_csv(dataset)