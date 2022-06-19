import random
from torchei import fault_model
from torchvision import models
import torch
import numpy as np
import sys
import os


random.seed(100)
np.random.seed(100)
torch.manual_seed(100)


def get_fi() -> fault_model:
    model = models.squeezenet1_0(pretrained=True)
    valid = torch.load("./datasets/ilsvrc_valid8.pt")[:1]
    return fault_model(model, valid)


def check_range(x, l=0.01, r=0.9, rate=1) -> bool:
    return x > l*rate and x < r*rate
