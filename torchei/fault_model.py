from copy import deepcopy
from functools import partial, wraps
from random import randint
from statistics import mean
from time import monotonic_ns
from types import MethodType
from typing import Any, Callable, List, TypeVar, Union
from datetime import datetime

import os
import logging
import traceback
import numpy as np
import torch
from tqdm import tqdm

from .utils import (
    emat,
    float_to_bin,
    get_result,
    monte_carlo,
    monte_carlo_hook,
    sequence_lim_adaptive,
    single_bit_flip,
    zscore_forward,
)

__all__ = ["fault_model", "cv_layer_filter", "nlp_layer_filter"]

cv_layer_filter = [
    ["weight"],  # must have
    ["feature", "conv", "fc", "linear", "classifier", "downsample", "att"],  # have one of
    ["bn"],  # don't have
    2,  # least dimension
]

nlp_layer_filter = [["weight"], 
                    ["embedding", "attention"], 
                    ["norm"], 
                    2]

data_type = TypeVar("data_type")
result_type = TypeVar("result_type")


def log_wrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(func.__name__,args,kwargs)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info(f"Locals : {locals()}")
            logging.error(f"Error in function {func.__name__}: {e}")
            traceback.print_exc()
            raise e
    return wrapper


class fault_model:
    """fault model of DNN in `torchei`"""

    @log_wrap
    def __init__(
        self,
        model: torch.nn.Module,
        input_data: data_type,
        infer_func: Callable[[data_type], result_type] = get_result,
        layer_filter: List = None,
        to_cuda: bool = True,
        device:str = "cuda:1"
    ) -> None:
        log_path = (
            os.path.dirname(os.path.abspath(__file__))
            + f"/log/{str(model)[:6]}{datetime.now().strftime('%m-%d-%H-%M')}.log"
        )
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s-%(asctime)s  : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, "a"),
            ],
        )
        print(f'log saved to {log_path}')
        logging.info(str(model)[:16])
        if layer_filter is None:
            layer_filter = cv_layer_filter
        model.eval()
        if to_cuda and torch.cuda.is_available():
            model.to(device)
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.to(device)
        self.model = model.requires_grad_(False)
        self.pure_dict = deepcopy(model.state_dict())
        example = [*self.pure_dict.values()][0]
        self.dtype = example.dtype
        self.device = example.device
        self.quant = self.dtype in [torch.qint8, torch.int8, torch.int8]
        self.cuda = example.is_cuda
        self.rng = np.random.Generator(np.random.PCG64DXSM())
        self.valid_data = input_data
        self.data_size = self.valid_data.shape[0] * self.valid_data.shape[1]
        self.infer = infer_func
        self._keys = []
        self.shape = []
        self.change_layer_filter(layer_filter)
        self.bitlen = 8 if self.quant else 32
        self.handles = []
        self.zero_rate = []
        self.input_shape = []
        self.register_hook(self.__save_layer_info_hook)
        self.ground_truth = infer_func(model, self.valid_data)
        self.clear_handles()
        self.compute_amount = []
        self.bit_dist = None
        self.synthesis_error = None
        self.p = None
        self.PerturbationTable = np.array(
            [-1, 2**128, 1 / (2**64), 1 / (2**32), 1 / (2**16), 10, 1],
            dtype=np.double,
        )
        self.PropTable = None
        # record time
        self.time = 0
        # this is a preempt var for user
        self.var = None

    @log_wrap
    def change_layer_filter(self, layer_filter: List[str]) -> None:
        """
        Select keys from state_dict according to layer_filter

        layer_filter consists of follow four elements:

        must contain all,must contain one of,can't contain,least dimensions of layer's weight:
        """
        assert len(layer_filter) == 4
        if not all(isinstance(x, list) for x in layer_filter[:-1]):
            raise AssertionError
        for key in self.get_all_keys():
            flag = [True, len(layer_filter[1]) == 0, True]
            for must_contain in layer_filter[0]:
                if must_contain not in key:
                    flag[0] = False
            for contain_one in layer_filter[1]:
                if contain_one in key:
                    flag[1] = True
                    break
            for dont_contain in layer_filter[2]:
                if dont_contain in key:
                    flag[2] = False
                    break
            if all(flag) and len(self.pure_dict[key].shape) >= layer_filter[-1]:
                self._keys.append(key)

        self.shape = [[*self.pure_dict[key].shape] for key in self._keys]
        self.layer_num = len(self.shape)

    def time_decorator(self, func):
        """return same function but record its time cost using self.time"""
        @wraps(func)
        def wrapper(*args, **kw):
            s = monotonic_ns()
            result = func(*args, **kw)
            e = monotonic_ns()
            self.time += (e - s) / 1e6
            return result

        return wrapper

    def weight_ei(self, inject_func) -> None:
        """low-level method to inject weight error"""
        corrupt_dict = deepcopy(self.pure_dict)
        inject_func(corrupt_dict, self.rng)
        self.model.load_state_dict(corrupt_dict)

    def neuron_ei(self, inject_hook: Callable[[torch.nn.Module, tuple], None]) -> None:
        """low-level method to inject neuron error"""
        self.clear_handles()
        self.register_hook(partial(inject_hook, self.rng), hook_type="forward_pre")

    @log_wrap
    def reliability_calc(
        self,
        iteration: int,
        error_inject: Callable[[None], None],
        kalman: bool = False,
        adaptive: bool = False,
        **kwargs,
    ) -> Union[List, float]:
        """Optional params:
        group_size:      Divide in group or not, >0 means group and its group_size
        kalman:          Use Kalman Filter in estimating
        adaptive:        Auto-Stop
        verbose_return:  return (estimation, group estimation, group index)
        """

        # all parameter should be exposed
        if kwargs.get("time_count", False):
            self.time = 0
            error_inject = self.time_decorator(error_inject)
        group_size = kwargs.get("group_size", 50)  # after change it to 0
        verbose_return = kwargs.get("verbose_return", False)
        group_estimation = []
        if adaptive:
            adaptive_func = kwargs.get("adaptive_func", sequence_lim_adaptive)
            if group_size <= 0:
                raise AssertionError
        if kalman:
            init_size = kwargs.get("init_size", 1000)
            if not (
                init_size % group_size == 0
                and init_size // group_size > 1
                and group_size > 1
            ):
                raise AssertionError
            error = 0
            for iter_times in tqdm(range(init_size)):
                error_inject()
                corrupt_result = self.infer(self.model, self.valid_data)
                error += torch.sum(corrupt_result != self.ground_truth)
                if (iter_times + 1) % group_size == 0:
                    group_estimation.append(error / self.data_size / group_size)
                    error = 0
                    # robust estimation 还没加上去
            mea_uncer_r, estimation_x = torch.var_mean(torch.tensor(group_estimation))
            est_uncert_p = self.get_param_size() * self.p / init_size / 32 / group_size
            estimation = [estimation_x]

        error = 0
        n = 0
        if locals().get("estimation") is None:
            estimation = [0]

        for iter_times in tqdm(range(iteration)):
            error_inject()
            corrupt_result = self.infer(self.model, self.valid_data)
            error += torch.sum(corrupt_result != self.ground_truth)
            if group_size and (iter_times + 1) % group_size == 0:
                n += 1
                z = error / self.data_size / group_size
                group_estimation.append(z)
                error = 0

                if kalman:
                    Kalman_Gain = est_uncert_p / (est_uncert_p + mea_uncer_r)
                    estimation.append(
                        estimation[-1] + Kalman_Gain * (z - estimation[-1])
                    )
                    est_uncert_p = est_uncert_p * (1 - Kalman_Gain)

                if adaptive:
                    if not kalman:
                        if n == 2:
                            estimation.pop(0)
                        estimation.append(
                            (n - 1) / n * estimation[-1] + 1 / n * group_estimation[-1]
                        )

                    if adaptive_func(estimation):
                        break

        if verbose_return:
            return estimation, group_estimation, n
        if adaptive or kalman:
            return estimation[-1].item()
        if n != 0:
            return torch.tensor(group_estimation).mean().item()
        return (error / self.data_size / iteration).item()

    def mc_attack(
        self,
        iteration: int,
        p: float,
        attack_func: Callable[[float], float] = single_bit_flip,
        kalman: bool = False,
        attack_type="weight",
        **kwargs,
    ) -> Union[List, float]:
        """Inject error using Monte Carlo method"""
        return self.reliability_calc(
            iteration=iteration,
            error_inject=self.get_mc_attacker(p, attack_func, attack_type),
            kalman=kalman,
            **kwargs,
        )

    def get_mc_attacker(
        self, p: float, attack_func: Callable[[float], float]=single_bit_flip, attack_type="weight"
    ) -> Callable[[None], None]:
        """Wrapper for injecting error using Monte Carlo method"""
        if attack_type == "weight":
            inject_func = partial(monte_carlo, attack_func, p, self._keys)
            error_inject = partial(self.weight_ei, inject_func)
        elif attack_type == "neuron":
            inject_func = partial(monte_carlo_hook, attack_func, p, self._keys)
            error_inject = partial(self.neuron_ei, inject_func)
        else:
            raise "Inject Type Error, you should select weight or neuron"
        self.p = p
        return error_inject

    def emat_attack(
        self,
        iteration: int,
        p: float,
        kalman: bool = False,
        **kwargs,
    ) -> Union[List, float]:
        """Inject error using EMAT method"""
        return self.reliability_calc(
            iteration=iteration,
            error_inject=self.get_emat_attacker(p),
            kalman=kalman,
            **kwargs,
        )

    def get_emat_attacker(self, p: float) -> Callable[[None], None]:
        """Wrapper for EMAT method, return a inject function"""
        p /= self.bitlen
        self.p = p
        self.__emat_calc()
        inject_func = partial(
            emat, self.PerturbationTable, self.PropTable, self.device, self._keys
        )
        return partial(self.weight_ei, inject_func)

    @log_wrap
    def layer_single_attack(
        self,
        layer_iter: int,
        attack_func: Callable[[float], Any] = None,
        layer_id: List = None,
        error_rate=True,
    ) -> List[float]:
        """Inject single error in layer per iteration"""
        if attack_func is None:
            attack_func = single_bit_flip
        result = []
        if layer_id is None:
            keys = self._keys
            shape = self.shape
        elif isinstance(layer_id, List):
            keys = [self._keys[idx] for idx in layer_id]
            shape = [self.shape[idx] for idx in layer_id]
        else:
            raise ("layer_id should be a list")
        for key_id, key in enumerate(keys):
            result.append([])
            for _ in tqdm(range(layer_iter)):
                corrupt_dict = deepcopy(self.pure_dict)
                corrupt_idx = tuple([randint(0, i - 1) for i in shape[key_id]])
                attack_result = attack_func(corrupt_dict[key][corrupt_idx].item())
                if not type(attack_result) is tuple:
                    corrupt_dict[key][corrupt_idx] = attack_result
                else:
                    if error_rate:
                        raise "If you need verbose info, you should calc error rate"
                    corrupt_dict[key][corrupt_idx] = attack_result[0]
                    result.append(attack_result[1:])
                self.model.load_state_dict(corrupt_dict)
                corrupt_result = self.infer(self.model, self.valid_data)
                result[key_id].append(
                    torch.sum(corrupt_result != self.ground_truth).item()
                )
        if error_rate:
            return [sum(i) / self.data_size / layer_iter for i in result]
        return result

    def get_layer_shape(self) -> List:
        return self.shape

    def get_param_size(self) -> int:
        """Calculate the total parameter size of the model"""
        param_size = 0
        for i in self.shape:
            temp = 1
            for j in i:
                temp *= j
            param_size += temp
        return param_size

    def get_all_keys(self) -> List[str]:
        return [*self.pure_dict.keys()]

    @log_wrap
    def calc_detail_info(self) -> None:
        """An auxiliary function for `sern_calc` to calculate the detail information of the model"""
        batch = self.valid_data.shape[0]
        layer_num = int(len(self.zero_rate) / batch)
        if batch != 1:
            temp = [
                mean([self.zero_rate[i + j * layer_num] for j in range(batch)])
                for i in range(layer_num)
            ]
            self.zero_rate = temp

        temp = [torch.tensor(self.valid_data.shape[2:]).tolist()]
        temp.extend(self.input_shape)
        self.input_shape = temp

        for i in range(layer_num):
            if len(self.shape[i]) == 4:
                s = self.shape[i][-1]
                n = self.input_shape[i][0]
                m = self.shape[i][0]
                self.compute_amount.append(
                    self.input_shape[i + 1][1] ** 2 * s * s * n * m
                )
            elif len(self.shape[i]) == 2:
                p = self.shape[i][0] * self.shape[i][1]
                self.compute_amount.append(p)
        self.clear_handles()

    def get_selected_keys(self) -> List[str]:
        return self._keys

    @log_wrap
    def sern_calc(self, output_class: int = None) -> List:
        """Calculating model's sbf error rate using sern algorithm"""
        if self.compute_amount == []:
            self.calc_detail_info()
        layernum = len(self.shape)
        nonzero = 1 - torch.tensor(self.zero_rate)
        sern = []
        input_size = (
            self.input_shape[0][0] * self.input_shape[0][1] * self.input_shape[0][2]
        )
        big_cnn = False
        k = 1 / 64
        if input_size > 200 * 200 * 3:
            big_cnn = True
        for i in range(layernum):
            later_compute = sum(self.compute_amount[i + 1 :])
            now_compute = later_compute + self.compute_amount[i]
            if i == layernum - 1:
                if len(self.shape[i]) != 2:
                    if output_class is None:
                        raise AssertionError
                    p_next = output_class
                else:
                    p_next = 1 / self.shape[i][0]
            else:
                p_next = nonzero[i + 1]
            if i == 0 and big_cnn:
                k = 1 / 32
            sern.append(
                k
                * (
                    nonzero[i]
                    * self.compute_amount[i]
                    * (1 + p_next)
                    / 2
                    / self.shape[i][0]
                    + later_compute
                )
                / now_compute
            )
            if i == 0 and big_cnn:
                k = 1 / 64
        return sern

    def unpack_weight(self) -> torch.Tensor:
        """Unpack the weight of the model to one tensor"""
        vessel = torch.tensor([])
        for i in self._keys:
            vessel = torch.cat((vessel, self.pure_dict[i].flatten().cpu()))
        return vessel

    def bit_distribution_statistic(self) -> List:
        """An auxiliary function for `emat_attack` to calculate the bit distribution of the model"""
        if self.bit_dist is None:
            weight = self.unpack_weight()
            weight = weight.numpy()
            self.rng.shuffle(weight)
            weight = torch.from_numpy(weight[:10000])
            bit_distri = torch.tensor([0 for i in range(self.bitlen)])
            for i in weight:
                bit = list(float_to_bin(i.item()))
                bit = [int(i) for i in bit]
                bit_distri += torch.tensor(bit)
            self.bit_dist = bit_distri / 10000.0
        return self.bit_dist

    def register_hook(self, hook: Callable[..., None], hook_type="forward") -> None:
        """Register a specified type hook function in specified layer"""
        model = self.model
        for key in self._keys:
            key = key.rsplit(".", 1)[0]
            module = model.get_submodule(key)
            if hook_type == "forward_pre":
                self.handles.append(module.register_forward_pre_hook(hook))
            elif hook_type == "forward":
                self.handles.append(module.register_forward_hook(hook=hook))

    def clear_handles(self) -> None:
        for i in self.handles:
            i.remove()
        self.handles = []

    @log_wrap
    def layer_alter(self, alter_func: Callable, layer_type, model=None) -> None:
        if model is None:
            model = self.model
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.layer_alter(alter_func, layer_type, module)

            if isinstance(module, layer_type):
                alter_func(model, module, name)

    @log_wrap
    def reluA_protection(self, protect_layers=torch.nn.ReLU) -> None:
        self.act_max = []

        def act_max_forward_hook(module, input, output):
            self.act_max.append(torch.max(output))

        def record_layer_actmax(model, relu: torch.nn.ReLU, name):
            relu.register_forward_hook(act_max_forward_hook)

        self.layer_alter(record_layer_actmax, torch.nn.ReLU)
        get_result(self.model, self.valid_data)
        self.clear_handles()
        self.var = 0

        def alter_reluA(model, _: torch.nn.ReLU, name) -> None:
            setattr(model, name, torch.nn.Hardtanh(0, self.act_max[self.var]))
            self.var += 1

        self.layer_alter(alter_reluA, protect_layers)

    @log_wrap
    def zscore_protect(self, layer_type:torch.nn.Module = torch.nn.Conv2d) -> None:
        """Use zscore detect bit flip errors"""
        model = self.model
        self.orig_model = deepcopy(model)
        self.config = []
        self.var = 0
        for key in self._keys:
            self.config.append(torch.std_mean(self.pure_dict[key]))

        def alter_zscore(model, layer: torch.nn.Module, name) -> None:
            layer.forward = MethodType(
                partial(zscore_forward, *self.config[self.var]),layer 
            )
            self.var += 1
            setattr(model, name, layer)

        self.layer_alter(alter_zscore, layer_type, model)
        self.var = 0
        self.model = torch.jit.trace(model, self.valid_data[0])
        get_result(self.model, self.valid_data)

    def zscore_protect_revoke(self) -> None:
        self.model = self.orig_model

    @log_wrap
    def relu6_protection(self, protect_layers=torch.nn.ReLU) -> None:
        """Warning:
        this will lower model's precision when no fault happening
        """
        self.layer_alter(
            lambda model, module, name: setattr(model, name, torch.nn.ReLU6()),
            protect_layers,
        )

    def __save_layer_info_hook(
        self, model: torch.nn.Module, input_val: torch.Tensor, output: torch.Tensor
    ) -> None:
        self.zero_rate.append(
            (torch.sum(input_val[0] == 0) / input_val[0].numel()).item()
        )
        self.input_shape.append(torch.tensor(output.shape[1:]).tolist())

    def __emat_calc(self) -> None:
        p = list(self.bit_distribution_statistic()[5:9])
        p.reverse()
        denom = 0
        for i in range(4):
            exp = 2 ** (2**i)
            denom += (1 / 2**exp) * p[i]
        self.synthesis_error = 4 / denom
        self.PerturbationTable[-2] = self.synthesis_error
        if self.p is not None:
            prop = np.array([1, 1, 1, 1, 1, 1])
            self.PropTable = np.append(prop * self.p, [1 - 6 * self.p])

    @log_wrap
    def get_emat_func(self) -> Callable[[float], float]:
        """return a simulate function that simulates single bit flip for ```layer single attack```"""
        self.__emat_calc()
        return (
            lambda num: num
            * torch.tensor(
                self.rng.choice(self.PerturbationTable[:-1]), dtype=torch.float64
            )
            if self.rng.random() < 6 / 32
            else num
        )
