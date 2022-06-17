from typing import OrderedDict,Callable
from random import randint
import struct
import torch
import numpy as np
from scipy import stats
__all__ = [
    "monte_carlo",
    "emat",
    "get_result",
    "sequence_lim_adaptive",
    "blank_hook",
    "zscore_dr_hook",
    "float_to_bin",
    "bin_to_float",
    "set_zero",
    "single_bit_flip_31",
    "single_bit_flip",
    "single_bit_flip_verbose",
    "twos_comp2int",
    "int2twos_comp",
    "dic_max",
    "monte_carlo_hook"
]

def emat(
    Per: list, Prop: list, device: str, keys: list, dic: OrderedDict, rng: np.random.Generator, *args
) -> None:
    for key in keys:
        inject_map = rng.choice(Per, dic[key].shape, p=Prop)
        dic[key] *= torch.tensor(inject_map, device=device)


def monte_carlo(
    attack_func: Callable[[float], float], p: float, keys: list, dic: OrderedDict, rng: np.random.Generator, *args
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
    prop: bool = False,
    sema=None,
)->torch.Tensor:
    if (not data.is_cuda) and next(model.parameters()).is_cuda:
        data = data.to("cuda")
    output = torch.tensor([], device=data.device)
    if sema is not None:
        sema.acquire()
    for i in data:
        output = torch.cat((output, model(i)))
    if sema is not None:
        sema.release()
    if prop:
        result = torch.zeros(output.shape[0], 2, topn)
    else:
        result = torch.zeros(output.shape[0], topn)
    for idx, i in enumerate(output):
        if prop:
            probabilities = torch.nn.functional.softmax(i, dim=0)
            topn_prop, topn_catid = torch.topk(probabilities, topn)
            result[idx][0] = topn_catid
            result[idx][1] = topn_prop
        else:
            topn_prop, topn_catid = torch.topk(i, topn)
            result[idx] = topn_catid
    return result


def sequence_lim_adaptive(estimation: list, times: int = 30, deviation: float = 0.01)->bool:
    if len(estimation) > times:
        for i in range(1, times + 1):
            if (
                torch.abs(
                    (estimation[-i] - estimation[-i - 1])) / estimation[-i - 1]
                > deviation
            ):
                return False
        return True
    else:
        return False


def blank_hook(module: torch.nn.Module, data: tuple, result: torch.tensor)->None:
    data = data[0]
    weight = module.weight
    weight_num = weight.numel()
    size = data.numel()
    pass

def zscore_dr_hook(module: torch.nn.Module, data: tuple)->None:
    data = data[0]
    for i in [module.weight, data]:
        if torch.max(i) > 2:
            value = i.to("cpu")
            trim = stats.trimboth(value, 0.1)
            mean = trim.mean()
            std = trim.std()
            zscore = torch.abs((value - mean) / std)
            outliers = [tuple(i) for i in torch.nonzero(zscore > 100000)]
            for idx in outliers:
                i[idx] = i[idx] / torch.tensor(2**128, dtype=torch.double)

def monte_carlo_hook(attack_func:Callable[[float],None], p:float, keys:list, rng: np.random.Generator ,module: torch.nn.Module, input:tuple)->None:
    input:torch.tensor = input[0].flatten()
    for i in range(len(input)):
        if rng.random()<p:
            input[i] = attack_func(input[i])
    

def float_to_bin(num: float)->list:
    return bin(struct.unpack("!I", struct.pack("!f", num))[0])[2:].zfill(32)


def bin_to_float(binary: list)->float:
    return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]


def set_zero(_)->int:
    return 0


def single_bit_flip_31(num: float)->float:
    return single_bit_flip(num, 1)


def single_bit_flip(num: float, bit: int = None, verbose: bool = False)->float:
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
        bits = bits[0:bit] + insert_bit + bits[bit + 1:]
        if verbose:
            return bin_to_float(bits), bit, insert_bit
        return bin_to_float(bits)
    else:
        raise TypeError("Error! You should input tensor or float!")


def single_bit_flip_verbose(num: float, bit: int = None)->float:
    return single_bit_flip(num, bit, verbose=True)


def twos_comp2int(n: int, bits: list)->list:
    s = bin(n & int("1" * bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)


def int2twos_comp(val: int, bits: list)->float:
    if (val & (1 << (bits - 1))) != 0:
        val = val - (1 << bits)
    return val


def pytorchfi_bit_flip(data: float, location: tuple):
    return single_bit_flip(data[location].item())


def pytorchfi_bit_flip_31(data: float, location: tuple):
    return single_bit_flip_31(data[location].item())


def dic_max(dic: OrderedDict)->float:
    return max([max(i) for i in dic.values()])
