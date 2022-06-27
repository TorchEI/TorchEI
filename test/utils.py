import random

import numpy as np
import torch
from torchvision import models

from torchei import fault_model

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)


def get_fi(name="squeeze") -> fault_model:
    model = models.mobilenet_v2(pretrained=True)
    valid = torch.load("data/ilsvrc_valid8.pt")[:1]
    return fault_model(model, valid)


def check_range(x, l_bound=0.01, r_bound=0.9, rate=1) -> bool:
    return l_bound * rate < x < r_bound * rate
