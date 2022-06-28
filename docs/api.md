# API Reference
<a id="torchei"></a>

## torchei

<a id="torchei.fault_model"></a>

## torchei.fault\_model

<a id="torchei.fault_model.fault_model"></a>

## fault\_model

```python
class fault_model()
```

fault model of DNN in `torchei`

<a id="torchei.fault_model.fault_model.change_layer_filter"></a>

### change\_layer\_filter

```python
def change_layer_filter(layer_filter: list) -> None
```

Select keys from state_dict according to layer_filter

layer_filter consists of follow four elements:

must contain all,must contain one of,can't contain,least dimensions of layer's weight:

<a id="torchei.fault_model.fault_model.time_decorator"></a>

### time\_decorator

```python
def time_decorator(func) -> Callable[..., Any]
```

return same function but record its time cost using self.time

<a id="torchei.fault_model.fault_model.reliability_calc"></a>

### reliability\_calc

```python
@torch.no_grad()
def reliability_calc(iteration: int,
                     error_inject: Callable[[None], None],
                     kalman: bool = False,
                     adaptive: bool = False,
                     **kwargs) -> Union[list, float]
```

Optional params:
group_size:      Divide in group or not, >0 means group and its group_size
kalman:          Use Kalman Filter in estimating
adaptive:        Auto-Stop
verbose_return:  return (estimation, group estimation, group index)

<a id="torchei.fault_model.fault_model.get_param_size"></a>

### get\_param\_size

```python
def get_param_size() -> int
```

Calculate the total parameter size of the model

<a id="torchei.fault_model.fault_model.calc_detail_info"></a>

### calc\_detail\_info

```python
@torch.no_grad()
def calc_detail_info() -> None
```

An auxiliary function for `sern_calc` to calculate the detail information of the model

<a id="torchei.fault_model.fault_model.sern_calc"></a>

### sern\_calc

```python
@torch.no_grad()
def sern_calc(output_class: int = None) -> list
```

Calculating model's sbf error rate using sern algorithm

<a id="torchei.fault_model.fault_model.unpack_weight"></a>

### unpack\_weight

```python
def unpack_weight() -> torch.Tensor
```

Unpack the weight of the model to one tensor

<a id="torchei.fault_model.fault_model.bit_distribution_statistic"></a>

### bit\_distribution\_statistic

```python
def bit_distribution_statistic() -> list
```

An auxiliary function for `emat_attack` to calculate the bit distribution of the model

<a id="torchei.fault_model.fault_model.register_hook"></a>

### register\_hook

```python
def register_hook(hook: Callable[..., None] = blank_hook,
                  hook_type="forward") -> None
```

Register a specified type hook function in specified layer

<a id="torchei.fault_model.fault_model.outlierDR_protection"></a>

### outlierDR\_protection

```python
def outlierDR_protection() -> None
```

Protect the model from bit flip errors

<a id="torchei.fault_model.fault_model.delimit"></a>

### delimit

```python
def delimit(num_points: int = 5,
            high: int = 100,
            interval: float = 0.5) -> list
```

return a list of points to delimit the certain range

<a id="torchei.fault_model.fault_model.relu6_protection"></a>

### relu6\_protection

```python
def relu6_protection(model: torch.nn.Module = None,
                     protect_layers=(torch.nn.ReLU)) -> None
```

**Warnings**:

  this will lower model's precision when no fault happening

<a id="torchei.fault_model.fault_model.get_emat_single_func"></a>

### get\_emat\_single\_func

```python
def get_emat_single_func() -> Callable[[float], float]
```

return a simulate function that simulates single bit flip for ```layer single attack```

<a id="torchei.fault_model.fault_model.stat_model"></a>

### stat\_model

```python
def stat_model(granularity: int = 1) -> None
```

Print the model's layer information include their keys

<a id="torchei.utils"></a>

## torchei.utils

<a id="torchei.utils.dic_max"></a>

### dic\_max

```python
def dic_max(dic: OrderedDict) -> float
```

return the max value of a dictionary

