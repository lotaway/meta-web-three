from nltk import pos_tag, download
from nltk.tokenize import word_tokenize
import torch
import platform
import importlib

# if not download('punkt_tab'):
#     print("Download punkt_tab not done")

def test():
    text = "NLP is a fascinating field of study."
    return tokenize(text)

def tokenize(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    print(pos_tags)
    return tokens

def get_device():
    device = None
    if platform.system() == 'Windows':
        torch_directml = importlib.import_module("torch_directml")
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