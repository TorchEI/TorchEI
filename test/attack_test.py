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
    
    def test_reliability_calc(self):
        fi = self.fi
        f = lambda _: False
        t = lambda _: True

        # test adaptive
        result = fi.emat_attack(10, 1e-5, kalman=True, group_return = True, group_size = 2, init_size = 4, adaptive = True, adaptive_func = f)
        self.assertEqual(result[-1],5)
        result = fi.emat_attack(10, 1e-5, kalman=True, group_return = True, group_size = 2, init_size = 4, adaptive = True, adaptive_func = t)
        self.assertEqual(result[-1],1)
        
        # test group_return
        result = fi.emat_attack(10, 1e-5, kalman=True, group_return = True, group_size = 2, init_size = 4)
        self.assertIsInstance(result,tuple)
        result = fi.emat_attack(10, 1e-5, kalman=True, group_return = False, group_size = 2, init_size = 4, adaptive = True)
        self.assertIsInstance(result,float)
        result = fi.emat_attack(10, 1e-5, kalman=False, group_return = True, group_size = 2, init_size = 4, adaptive = True)
        self.assertIsInstance(result,tuple)

        #test group_size
        result = fi.emat_attack(10, 1e-5, group_size = None)
        self.assertIsInstance(result,float)
    
    def test_neuron_ei(self):
        result = self.fi.mc_attack(10,1e-5,type="neuron")
        self.assertTrue(check_range(result))