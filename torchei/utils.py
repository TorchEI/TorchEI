import struct
from random import randint
from typing import Callable, OrderedDict, List

import numpy as np
import torch

__all__ = [
    "monte_carlo",
    "emat",
    "get_result",
    "sequence_lim_adaptive",
    "float_to_bin",
    "bin_to_float",
    "single_bit_flip_31",
    "single_bit_flip",
    "single_bit_flip_verbose",
    "monte_carlo_hook",
    "zscore_forward",
]


def zscore_forward(
    std: torch.Tensor(), mean: torch.Tensor, self: torch.nn.Conv2d, x: torch.Tensor
):
    self.weight *= ~(torch.abs((self.weight - mean) / std) > 1000)
    return self._conv_forward(x, self.weight, self.bias)


def emat(
    Per: List[float],
    Prop: List[float],
    device: str,
    keys: List[str],
    dic: OrderedDict,
    rng: np.random.Generator,
    *args
) -> None:
    for key in keys:
        inject_map = rng.choice(Per, dic[key].shape, p=Prop)
        dic[key] *= torch.tensor(inject_map, device=device)


def monte_carlo(
    attack_func: Callable[[float], float],
    p: float,
    keys: List[str],
    dic: OrderedDict,
    rng: np.random.Generator,
    *args
) -> None:
    for key in keys:
        weight_matrix = dic[key].flatten()
        for weight_idx in range(len(weight_matrix)):
            if rng.random() < p:
                weight_matrix[weight_idx] = attack_func(
                    weight_matrix[weight_idx].item()
                )


@torch.no_grad()
def get_result(
    model: torch.nn.Module,
    data: torch.Tensor,
    topn: int = 1,
    reserve_prob: bool = False, # return probability of output
) -> torch.Tensor:
    if (not data.is_cuda) and next(model.parameters()).is_cuda:
        data = data.to(next(model.parameters()))
    output = torch.tensor([], device=data.device)
    for i in data:
        output = torch.cat((output, model(i)))
    if reserve_prob:
        result = torch.zeros(output.shape[0], 2, topn)
    else:
        result = torch.zeros(output.shape[0], topn)
    for idx, i in enumerate(output):
        if reserve_prob:
            probabilities = torch.nn.functional.softmax(i, dim=0)
            topn_prop, topn_catid = torch.topk(probabilities, topn)
            result[idx][0] = topn_catid
            result[idx][1] = topn_prop
        else:
            _, topn_catid = torch.topk(i, topn)
            result[idx] = topn_catid
    return result


def sequence_lim_adaptive(
    estimation: List, times: int = 30, deviation: float = 0.01
) -> bool:
    if len(estimation) > times:
        return all(
            not (
                torch.abs((estimation[-i] - estimation[-i - 1])) / estimation[-i - 1]
                > deviation
            )
            for i in range(1, times + 1)
        )
    return False


def monte_carlo_hook(
    attack_func: Callable[[float], None],
    p: float,
    keys: List[str],
    rng: np.random.Generator,
    module: torch.nn.Module,
    input_data: tuple,
) -> None:
    input_data: torch.Tensor = input_data[0].flatten()
    for i in range(len(input_data)):
        if rng.random() < p:
            input_data[i] = attack_func(input_data[i])


def float_to_bin(num: float) -> List:
    return bin(struct.unpack("!I", struct.pack("!f", num))[0])[2:].zfill(32)


def bin_to_float(binary: List) -> float:
    return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]


def set_zero(_) -> int:
    return 0


def single_bit_flip_31(num: float) -> float:
    return single_bit_flip(num, 1)


def single_bit_flip(num: float, bit: int = None, verbose: bool = False) -> float:
    if bit is None:
        bit = randint(0, 31)
    if isinstance(num, torch.Tensor):
        num = num.item()
    if isinstance(num, float):
        bits = float_to_bin(num)
        if bits[bit] == "1":
            insert_bit = "0"
        elif bits[bit] == "0":
            insert_bit = "1"
        else:
            print("Error !!!")
        bits = bits[0:bit] + insert_bit + bits[bit + 1 :]
        if verbose:
            return bin_to_float(bits), bit, insert_bit
        return bin_to_float(bits)
    raise TypeError("Error! You should input tensor or float!")


def single_bit_flip_verbose(num: float, bit: int = None) -> float:
    return single_bit_flip(num, bit, verbose=True)
