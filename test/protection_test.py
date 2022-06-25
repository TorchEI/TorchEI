import random
import unittest

import numpy as np
import torch

from .utils import get_fi

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)


class ProtectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fi = get_fi()
        self.unprotected = self.fi.emat_attack(10, 1e-5)
        return super().setUp()

    def test_relu6(self):
        self.fi.relu6_protection()
        protected = self.fi.emat_attack(10, 1e-5)
        print(protected)
        print(self.unprotected)
        self.assertTrue(protected != self.unprotected)

    def test_zscoreDR(self):
        self.fi.outlierDR_protection()
        protected = self.fi.emat_attack(10, 1e-5)
        self.assertTrue(protected < self.unprotected * 0.7)
