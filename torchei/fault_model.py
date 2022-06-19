from copy import deepcopy
import logging
from math import log10 as lg
import random
from statistics import mean
from time import monotonic_ns
from typing import Any, Callable, OrderedDict, TypeVar, Union
import numpy as np
import torch
from tqdm import tqdm
import torchstat
from functools import partial
from .utils import *
from random import randint
from .utils import monte_carlo_hook

__all__ = ["fault_model"]

default_layer_filter = [
    ["weight"],
    ["feature", "conv", "fc", "linear", "classifier", "downsample"],
    ["bn"],
    2,
]

data_type = TypeVar("data_type")
result_type = TypeVar("result_type")


class fault_model:
    @torch.no_grad()
    def __init__(
        self,
        model: torch.nn.Module,
        input_data: data_type,
        infer_func: Callable[[data_type], result_type] = get_result,
        layer_filter: list = default_layer_filter,
        to_cuda: bool = True,
    ) -> None:
        model.eval()
        if to_cuda and torch.cuda.is_available():
            model.to("cuda")
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.to("cuda")
        self.model = model
        self.pure_dict = deepcopy(model.state_dict())
        example = [*self.pure_dict.values()][0]
        self.dtype = example.dtype
        self.device = example.device
        self.quant = (
            True if self.dtype in [torch.qint8,
                                   torch.int8, torch.int8] else False
        )
        self.cuda = example.is_cuda
        self.rng = np.random.Generator(np.random.PCG64DXSM())
        self.valid_data = input_data
        self.data_size = self.valid_data.shape[0] * self.valid_data.shape[1]
        self.infer = infer_func
        self.keys = []
        self.bitlen = 8 if self.quant else 32
        self.ground_truth = infer_func(model, self.valid_data)
        self.shapes = None
        self.zero_rate = []
        self.input_shape = []
        self.compute_amount = []
        self.handles = []
        self.bit_dist = None
        self.synthesis_error = None
        self.p = None
        self.layer_num = 0
        self.change_layer_filter(layer_filter)
        self.PerturbationTable = np.array(
            [-1, 2**128, 1 / (2**64), 1 / (2**32), 1 / (2**16), 10, 1],
            dtype=np.double,
        )
        self.PropTable = None
        # record time
        self.time = 0
        # this is a preempt var for user
        self.test_var = None

    def change_layer_filter(self, layer_filter: list) -> None:
        for key in [*self.pure_dict.keys()]:
            for pos_filter in layer_filter[0]:
                if pos_filter in key:
                    for type_filter in layer_filter[1]:
                        if type_filter in key:
                            for dis_filter in layer_filter[2]:
                                if dis_filter not in key:
                                    if (
                                        len(self.pure_dict[key].size())
                                        >= layer_filter[-1]
                                    ):
                                        self.keys.append(key)
                                    break
                            break
                break

        self.shapes = [[*self.pure_dict[key].shape] for key in self.keys]
        self.layer_num = len(self.shapes)

    def time_decorator(self, func) -> Callable[..., Any]:
        def wrapper(*args, **kw):
            s = monotonic_ns()
            result = func(*args, **kw)
            e = monotonic_ns()
            self.time += (e - s) / 1e6
            return result

        return wrapper

    def weight_ei(self, inject_func) -> None:
        corrupt_dict = deepcopy(self.pure_dict)
        inject_func(corrupt_dict, self.rng)
        self.model.load_state_dict(corrupt_dict)

    def neuron_ei(self, inject_hook: Callable[[torch.nn.Module, tuple], None]) -> None:
        self.clear_handles()
        self.register_hook(partial(inject_hook, self.rng), type="forward_pre")

    @torch.no_grad()
    def reliability_calc(
        self,
        iteration: int,
        error_inject: Callable[[None], None],
        kalman: bool = False,
        adaptive: bool = False,
        **kwargs,
    ) -> Union[list, float]:
        """
        group_size:      Divide in group or not, >0 means group and its group_size 
        kalman:          Use Kalman Filter in estimating
        adaptive:        Auto-Stop       
        verbose_return:  Return a tuple of estimation, group estimation and group index or just latest estimation
        """
        try:
            # all parameter should be exposed
            if kwargs.get("time_count", True):
                self.time = 0
                error_inject = self.time_decorator(error_inject)
            group_size = kwargs.get("group_size", 50)  # after change it to 0
            verbose_return = kwargs.get("verbose_return", False)
            group_estimation = []
            if adaptive:
                adaptive_func = kwargs.get(
                    "adaptive_func", sequence_lim_adaptive)
                assert(group_size > 0)
            if kalman:
                init_size = kwargs.get("init_size", 1000)
                assert (init_size % group_size ==0 and init_size//group_size >1 and group_size > 1 )
                error = 0
                for iter in tqdm(range(init_size)):
                    error_inject()
                    corrupt_result = self.infer(self.model, self.valid_data)
                    error += torch.sum(corrupt_result != self.ground_truth)
                    if (iter + 1) % group_size == 0:
                        group_estimation.append(
                            error / self.data_size / group_size)
                        error = 0
                        # robust estimation 还没加上去
                mea_uncer_r, estimation_x = torch.var_mean(
                    torch.tensor(group_estimation))
                est_uncert_p = (
                    self.get_param_size() * self.p / init_size / 32 / group_size
                )
                estimation = [estimation_x]

            error = 0
            n = 0
            if locals().get('estimation') is None:
                estimation = [0]

            for iter in tqdm(range(iteration)):
                error_inject()
                corrupt_result = self.infer(self.model, self.valid_data)
                error += torch.sum(corrupt_result != self.ground_truth)
                if group_size and (iter + 1) % group_size == 0:
                    n += 1
                    z = error / self.data_size / group_size
                    group_estimation.append(z)
                    error = 0

                    if kalman:
                        Kalman_Gain = est_uncert_p / \
                            (est_uncert_p + mea_uncer_r)
                        estimation.append(
                            estimation[-1] + Kalman_Gain * (z - estimation[-1]))
                        est_uncert_p = est_uncert_p * (1 - Kalman_Gain)

                    if adaptive:
                        if not kalman:
                            if(n == 2):
                                estimation.pop(0)
                            estimation.append(
                                (n - 1) / n * estimation[-1] +
                                1 / n * group_estimation[-1]
                            )

                        if adaptive_func(estimation):
                            break

            if verbose_return:
                return estimation, group_estimation, n
            elif adaptive or kalman:
                return estimation[-1].item()
            elif n!=0:
                return torch.tensor(group_estimation).mean()
            else:
                return (error/self.data_size/iteration).item()

        except Exception as e:
            logging.error(f"error happened while calc reliability\n{e}")
            last_estimation = estimation[-1]
            logging.log(5,
                f"Unsaved values:group\ngroup_estimation:{group_estimation}\nestimation:{last_estimation}\niterTimes{n}")
            print(f"error happened while calc reliability\n{e}")

    # merge two attack method
    def mc_attack(
        self,
        iteration: int,
        p: float,
        attack_func: Callable[[float], float] = None,
        kalman: bool = False,
        type="weight",
        **kwargs,
    ) -> Union[list, float]:
        if attack_func is None:
            attack_func = single_bit_flip
        if type == "weight":
            inject_func = partial(monte_carlo, attack_func, p, self.keys)
            error_inject = partial(self.weight_ei, inject_func)
        elif type == "neuron":
            inject_func = partial(monte_carlo_hook,
                                  attack_func, p, self.keys)
            error_inject = partial(self.neuron_ei, inject_func)
        else:
            raise("Inject Type Error, you should select weight or neuron")
        self.p = p
        return self.reliability_calc(
            iteration=iteration,
            error_inject=error_inject,
            kalman=kalman,
            **kwargs,
        )

    def emat_attack(
        self,
        iteration: int,
        p: float,
        kalman: bool = False,
        type="weight",
        **kwargs,
    ) -> Union[list, float]:
        p /= self.bitlen
        self.p = p
        self.__emat_calc()
        if type != "weight":
            raise("Inject Type Error, emat only support attack on weight")
        inject_func = partial(
            emat, self.PerturbationTable, self.PropTable, self.device, self.keys
        )
        return self.reliability_calc(
            iteration=iteration,
            error_inject=partial(self.weight_ei, inject_func),
            kalman=kalman,
            **kwargs,
        )

    @torch.no_grad()
    def layer_single_attack(
        self, layer_iter: int, attack_func: Callable[[float], Any] = None, error_rate = True
    ) -> list:
        if attack_func is None:
            attack_func = single_bit_flip
        result = []
        for key_id, key in enumerate(self.keys):
            result.append([])
            for _ in tqdm(range(layer_iter)):
                corrupt_dict = deepcopy(self.pure_dict)
                corrupt_idx = tuple([randint(0, i - 1)
                                    for i in self.shapes[key_id]])
                attack_result = attack_func(
                    corrupt_dict[key][corrupt_idx].item())
                if not (type(attack_result) is tuple):
                    corrupt_dict[key][corrupt_idx] = attack_result
                else:
                    if error_rate:
                        raise("If you need verbose info, you should calc error rate yourself")
                    corrupt_dict[key][corrupt_idx] = attack_result[0]
                    result.append(attack_result[1:])
                self.model.load_state_dict(corrupt_dict)
                corrupt_result = self.infer(self.model, self.valid_data)
                result[key_id].append(torch.sum(corrupt_result !=
                                                self.ground_truth).item())
        if error_rate:
            return [sum(i)/self.data_size/layer_iter for i in result]
        return result

    def get_layer_shape(self) -> list:
        return self.shapes

    def get_param_size(self) -> int:
        param_size = 0
        for i in self.shapes:
            temp = 1
            for j in i:
                temp *= j
            param_size += temp
        return param_size

    def get_all_keys(self) -> list:
        return [*self.pure_dict.keys()]

    @torch.no_grad()
    def calc_detail_info(self) -> None:
        self.register_hook(self.__save_layer_info_hook)
        self.infer(self.model, self.valid_data)

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
            if len(self.shapes[i]) == 4:
                s = self.shapes[i][-1]
                n = self.input_shape[i][0]
                m = self.shapes[i][0]
                self.compute_amount.append(
                    self.input_shape[i + 1][1] ** 2 * s * s * n * m
                )
            elif len(self.shapes[i]) == 2:
                p = self.input_shape[i][0]
                assert p == self.shapes[i][1]
                self.compute_amount.append(p)
        self.clear_handles()

    def get_selected_keys(self) -> list:
        return self.keys

    @torch.no_grad()
    def sern_calc(self, output_class: int = None) -> list:
        if self.compute_amount == []:
            self.calc_detail_info()
        layernum = len(self.shapes)
        nonzero = 1 - torch.tensor(self.zero_rate)
        sern = []
        input_size = (
            self.input_shape[0][0] *
            self.input_shape[0][1] * self.input_shape[0][2]
        )
        big_cnn = False
        k = 1 / 64
        if input_size > 200 * 200 * 3:
            big_cnn = True
        for i in range(layernum):
            later_compute = sum(self.compute_amount[i + 1:])
            now_compute = later_compute + self.compute_amount[i]
            if i == layernum - 1:
                if len(self.shapes[i]) != 2:
                    assert output_class != None
                    p_next = output_class
                else:
                    p_next = 1 / self.shapes[i][0]
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
                    / self.shapes[i][0]
                    + later_compute
                )
                / now_compute
            )
            if i == 0 and big_cnn:
                k = 1 / 64
        return sern

    def unpack_weight(self) -> torch.Tensor:
        vessel = torch.tensor([])
        for i in self.keys:
            vessel = torch.cat((vessel, self.pure_dict[i].flatten().cpu()))
        return vessel

    def bit_distribution_statistic(self) -> list:
        if self.bit_dist is None:
            weight = self.unpack_weight()
            weight = weight.numpy()
            np.random.shuffle(weight)
            weight = torch.from_numpy(weight[:10000])
            bit_distri = torch.tensor([0 for i in range(self.bitlen)])
            for i in weight:
                bit = list(float_to_bin(i.item()))
                bit = [int(i) for i in bit]
                bit_distri += torch.tensor(bit)
            self.bit_dist = bit_distri / 10000.0
        return self.bit_dist

    def register_hook(
        self, hook: Callable[..., None] = blank_hook, type="forward"
    ) -> None:
        model = self.model
        for key in self.keys:
            key = key.rsplit(".", 1)[0]
            module = model.get_submodule(key)
            if type == "forward_pre":
                self.handles.append(
                    module.register_forward_pre_hook(hook=hook))
            elif type == "forward":
                self.handles.append(module.register_forward_hook(hook=hook))

    def outlierDR_protection(self) -> None:
        self.register_hook(zscore_dr_hook, type="forward_pre")

    def clear_handles(self) -> None:
        for i in self.handles:
            i.remove()
        self.handles = []

    def delimit(
        self, num_points: int = 5, high: int = 100, interval: float = 0.5
    ) -> list:
        param_size = self.get_param_size()
        max_point = np.double(-lg((high / param_size)))
        manti = max_point % 0.5
        max_point = max_point - manti
        if manti > 0.3:
            max_point += 0.5
        if max_point % 1 >= 0.49:
            max_point += 0.5
        points = []
        for i in range(num_points):
            points.append(max_point + i * interval)
        return points

    def relu6_protection(self, model: torch.nn.Module = None) -> None:
        if model is None:
            model = self.model
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                self.relu6_protection(module)

            if isinstance(module, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh)):
                setattr(model, n, torch.nn.ReLU6())

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
            exp = 2**i
            denom += (1 / 2**exp) * p[i]
        self.synthesis_error = 4 / denom
        self.PerturbationTable[-2] = self.synthesis_error
        if self.p is not None:
            prop = np.array([1, 1, 1, 1, 1, 1])
            self.PropTable = np.append(prop * self.p, [1 - 6 * self.p])

    def get_emat_single_func(self)->Callable[[float],float]:
        self.__emat_calc()
        return lambda num : num*torch.tensor(random.choice(self.PerturbationTable[:-1]),dtype = torch.float64) if random.random() < 6/32 else num


    def stat_model(self, granularity: int = 1) -> None:
        torchstat.stat(self.model, self.input_shape, granularity)
