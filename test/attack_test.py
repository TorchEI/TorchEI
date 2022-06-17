from statistics import mean
import torch
from utils import get_fi, check_range
import numpy as np
import unittest
import random

a = [1,2]

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)


class AttackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fi = get_fi()
        return super().setUp()

    def test_emat(self):
        result = self.fi.emat_attack(10, 1e-5)
        self.assertTrue(check_range(result))

    def test_mc(self):
        result = self.fi.mc_attack(10, 1e-5)
        self.assertTrue(check_range(result))

    def test_layer_single_attack(self):
        result = self.fi.layer_single_attack(100)
        for i in result[:3]:
            self.assertTrue(check_range(mean(i)))
    
    def test_weightei_kalman(self):
        # 1. kalman
        # 2. group
        # 3. groupreturnsdasdasdkljDowners Grove
        # 4. no group
        # 5. group adaptive
        return 1
    
    def test_neuron_ei(self):
        result = self.fi.mc_attack(10,1e-5,type="neuron")
        self.assertTrue(check_range(result))