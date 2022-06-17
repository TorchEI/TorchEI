<div align="center">
  <img src="https://raw.githubusercontent.com/TorchEI/TorchEI/main/assets/torchei.svg" alt="torchei_logo" align="center" style="width:30%;"  />
</div>


<h1 style = "margin:0;" align="center">TorchEI⚡</h1>

<div align = "center" style="font-weight: bold;"><a href="#introduction">Intro</a> ● <a href="#quick-example">Usage</a> ● <a href="https://TorchEI.github.com.io/TorchEI/">Doc</a>  ● <a href="#citation">Cite</a> ● <a href="#contribution">Contribution</a> ● <a href="#license">License</a></div>


## Introduction


👋TorchEI, pronouced */ˈtôrCHər/*, short for Pytorch Error Injection, is a high-speed toolbox around DNN Reliability's Research and Development. TorchEI enables you quickly and simply inject errors into DNN, collects information you needed, and harden your DNN.

TorchEI implemented incredible parallel evaluation system which could allow you adequately utilize device computing performance with tolerance to non-catastrophic faults.

## Features

- Full typing system supported
- Contains methods from papers in DNN Reliability
- High-efficiency, fault-tolerant parallel system

## Quick Example

Here we gonna show you a quick example, or you can try [interactive demo](https://colab.research.google.com/github/TorchEI/TorchEI/blob/main/example.ipynb) and [online edtior](https://github.dev/TorchEI/TorchEI).

#### Installing

Install public distribution using  `pip3 install torchei` or [download](https://github.com/TorchEI/TorchEI/archive/refs/heads/main.zip) it.

#### Example

Init fault model

```python
import torch
from torchvision import models
import torchei
model = models.resnet18(pretrained=True)
data = torch.load('./datasets/ilsvrc_valid8.pt')
fault_model = torchei.fault_model(model,data)
```

Calc reliability using emat method

```python
fault_model.emat_attack(10,1e-3)
```



Calc reliability using Parallel Mechanism (under developing)

```python

```



Calc reliability using [SERN](https://dl.acm.org/doi/abs/10.1145/3386263.3406938) 

```python
fault_model.sern_calc(output_class=1000)
```


Harden DNN by ODR

```python
fault_model.outlierDR_protection()
fault_model.emat_attack(10,1e-3)
```

## Contribution
If you find🧐 any bugs or have🖐️ any suggestion, please tell us.

This repo is open to everyone wants to maintain together.

You can helps us with follow things:
- PR your implemented methods in your or others' papers
- Complete our project
- Translate our docs to your language
- Other

We want to build TorchEI to best toolbox in DNN Reliability around bit flip, adversarial attack, and others. 
:e-mail: forcessless@foxmail.com

## Citation

Our paper is under reviewing.

## License
> [MIT](./LICENSE) License.
> Copyright:copyright:2022/5/23-present, Hao Zheng.