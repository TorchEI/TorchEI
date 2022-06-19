import torch
from .utils import get_fi, check_range
import numpy as np
import unittest
import random
from time import sleep


random.seed(100)
np.random.seed(100)
torch.manual_seed(100)


class ToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fi = get_fi()
        return super().setUp()

    def test_sern(self):
        sern = self.fi.sern_calc(1000)
        self.assertTrue(check_range(sern[0],0.027,0.029))
    
    def test_time_decorator(self):
        @self.fi.time_decorator
        def hibernate(x):
            sleep(x)
        span = random.randrange(0,5)
        hibernate(span)
        self.assertTrue(check_range(self.fi.time,span-0.1,span+0.1,1000))
    
    def test_change_layer_filter(self):
        return 1

    def test_synthesis_error(self):
        self.fi.get_emat_single_func()
        self.assertTrue(check_range(self.fi.synthesis_error,8,12))
    
    def test_register_hook(self):
        self.fi.register_hook()
        self.assertEqual(len(self.fi.handles),len(self.fi.keys))