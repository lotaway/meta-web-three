import torch
import torch_directml

# check if GPU device work
# if torch.backends.mps.is_available(): # apple silicon
# if torch.cuda.is_available(): # nvidia
# if torch.backends.hip.is_available(): # amd
#     # device = torch.device("mps")
#     device = torch.device("mps") # use AMD Metal Performance Shaders
#     print("AMD GPU is available.")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, using CPU.")

if(torch.cuda.is_available()):
    print("cuda is available")
else:
    print("cuda is not available")

device = torch_directml.device()
tensor = torch.randn(2, 2).to(device)
print(tensor)