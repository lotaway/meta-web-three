import torch

# check if GPU device work
if torch.backends.mps.is_available():
    device = torch.device("mps") # use AMD Metal Performance Shaders
    print("AMD GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")