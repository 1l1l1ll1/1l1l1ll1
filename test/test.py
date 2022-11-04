import FC as model
import torch

for name, param in model.named_parameters():
    print(name,param)