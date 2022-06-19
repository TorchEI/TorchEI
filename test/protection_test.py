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
        self.unprotected = self.fi.emat_attack(200,1e-5)
        return super().setUp()

    def test_relu6(self):
        self.fi.relu6_protection()
        protected = self.fi.emat_attack(200,1e-6)
        self.assertTrue(protected<self.unprotected*0.9)

    def test_zscoreDR(self):
        self.fi.outlierDR_protection()
        protected = self.fi.emat_attack(200,1e-6)
        self.assertTrue(protected<self.unprotected*0.6)
