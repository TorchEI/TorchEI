import random
import unittest
from time import sleep

import numpy as np
import torch

from .utils import check_range, get_fi

random.seed(100)
np.random.seed(100)
torch.manual_seed(100)


class ToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fi = get_fi()
        return super().setUp()

    def test_sern(self):
        sern = self.fi.sern_calc(1000)
        self.assertTrue(check_range(sern[0], 0.02, 0.04))

    def test_time_decorator(self):
        @self.fi.time_decorator
        def hibernate(x):
            sleep(x)

        span = random.randrange(0, 5)
        hibernate(span)
        self.assertTrue(check_range(self.fi.time, span - 0.1, span + 0.1, 1000))

    def test_change_layer_filter(self):
        return 1

    def test_synthesis_error(self):
        self.fi.get_emat_func()
        self.assertTrue(check_range(self.fi.synthesis_error, 4, 32))

    def test_register_hook(self):
        def blank_hook(x, y):
            pass

        self.fi.register_hook(blank_hook)
        self.assertEqual(len(self.fi.handles), len(self.fi.keys))

    def test_delimit(self):
        limit_points = self.fi.delimit()
        param_size = self.fi.get_param_size()
        self.assertAlmostEqual(10 ** (-limit_points[0]) * param_size, 100, delta=50)
