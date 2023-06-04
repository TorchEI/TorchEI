import random
import unittest
from statistics import mean

import numpy as np
import torch

from .utils import check_range, get_fi

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
        result = self.fi.layer_single_attack(10)
        self.assertTrue(check_range(mean(result), 1 / 64 / 2, 0.1))
        attack_func = self.fi.get_emat_func()
        self.fi.layer_single_attack(10, attack_func=attack_func)
        self.assertTrue(check_range(mean(result), 1 / 64 / 2, 0.1))

    def test_reliability_calc(self):
        fi = self.fi

        def f(_):
            return False

        def t(_):
            return True

        result = fi.emat_attack(
            10,
            1e-5,
            group_size=2,
            adaptive=True,
            adaptive_func=f,
        )
        self.assertIsInstance(result, float)

        # test adaptive
        result = fi.emat_attack(
            10,
            1e-5,
            kalman=True,
            verbose_return=True,
            group_size=2,
            init_size=4,
            adaptive=True,
            adaptive_func=f,
        )
        self.assertEqual(result[-1], 5)
        result = fi.emat_attack(
            10,
            1e-5,
            kalman=True,
            verbose_return=True,
            group_size=2,
            init_size=4,
            adaptive=True,
            adaptive_func=t,
        )
        self.assertEqual(result[-1], 1)

        # test verbose_return
        result = fi.emat_attack(
            10, 1e-5, kalman=True, verbose_return=True, group_size=2, init_size=4
        )
        self.assertIsInstance(result, tuple)
        result = fi.emat_attack(
            10,
            1e-5,
            kalman=True,
            verbose_return=False,
            group_size=2,
            init_size=4,
            adaptive=True,
        )
        self.assertIsInstance(result, float)
        result = fi.emat_attack(
            10,
            1e-5,
            kalman=False,
            verbose_return=True,
            group_size=2,
            init_size=4,
            adaptive=True,
        )
        self.assertIsInstance(result, tuple)

    def test_neuron_ei(self):
        result = self.fi.mc_attack(10, 1e-5, attack_type="neuron")
        self.assertTrue(check_range(result))
