import torch
from .utils import get_fi
import numpy as np
import unittest
import random

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)


class ProtectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fi = get_fi()
        self.unprotected = self.fi.emat_attack(100,1e-3)
        return super().setUp()

    def test_relu6(self):
        self.fi.relu6_protection()
        protected = self.fi.emat_attack(100,1e-3)
        print(protected)
        print(self.unprotected)
        self.assertTrue(protected<self.unprotected)

    def test_zscoreDR(self):
        self.fi.outlierDR_protection()
        protected = self.fi.emat_attack(100,1e-3)
        self.assertTrue(protected<self.unprotected*0.7)
