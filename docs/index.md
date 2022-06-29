<div align="center">
  <img src="https://raw.githubusercontent.com/TorchEI/TorchEI/main/assets/torchei.svg" alt="torchei_logo" align="center" style="width:30%;"  />
</div>

<h1 style = "margin:0;" align="center">TorchEI⚡</h1>

<div align = "center" style="font-weight: bold;"><a href="#introduction">Intro</a> ● <a href="#quick-example">Usage</a> ● <a href="https://TorchEI.github.com.io/TorchEI/">Doc</a>  ● <a href="#citation">Cite</a> ● <a href="#contribution" >Contribution</a> ● <a href="#license">License</a></div>

---

<div align = "center">
    <a href = "https://github.com/TorchEI/TorchEI/actions/workflows/pytest-cov.yml">
  <img src="https://github.com/TorchEI/TorchEI/actions/workflows/pytest-cov.yml/badge.svg"/></a>
 <a href="https://codecov.io/gh/TorchEI/TorchEI" >
   <img src="https://codecov.io/gh/TorchEI/TorchEI/branch/main/graph/badge.svg?token=0ADLQFHLCJ"/></a>
 <a href="https://www.codacy.com/gh/TorchEI/TorchEI/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=TorchEI/TorchEI&amp;utm_campaign=Badge_Grade">
  <img src="https://app.codacy.com/project/badge/Grade/c4067d004b934d49bb4386b650c57808"/></a>
 <a href="https://pypi.org/project/torchei/"  target=”_blank”>
    <img src="https://img.shields.io/pypi/v/torchei" alt="Pypi"></a>
    <a href="https://pypi.org/project/torchei/"  target=”_blank”>
     <img src="https://pepy.tech/badge/torchei"/></a>
 <a href="#license">
    <img src="https://img.shields.io/github/license/torchei/torchei" alt="License"></a>
</div>

## Introduction

👋TorchEI, pronounced*/ˈtôrCHər/*, short for Pytorch Error Injection, is a high-speed toolbox for DNN Reliability's Research and Development. TorchEI enables you quickly and simply inject errors into DNN, collects information you needed, and harden your DNN.

TorchEI implemented incredible parallel evaluation system which could allow you adequately utilize device computing performance with tolerance to non-catastrophic faults.

## Features

- Full typing system supported
- Contains methods from papers in DNN Reliability
- High-efficiency, fault-tolerant parallel system

## Quick Example

Here we gonna show you a quick example, or you can try [interactive demo](https://colab.research.google.com/github/TorchEI/TorchEI/blob/main/example.ipynb) and [online editor](https://github.dev/TorchEI/TorchEI).

#### Installing

Install public distribution using `pip3 install -U torchei -i https://pypi.org/simple` or [download](https://github.com/TorchEI/TorchEI/archive/refs/heads/main.zip) it.

#### Example

Init fault model

```python
import torch
from torchvision import models
import torchei
model = models.resnet18(pretrained=True)
data = torch.load('data/ilsvrc_valid8.pt')
fault_model = torchei.fault_model(model,data)
```

Calc reliability using emat method

```python
fault_model.emat_attack(10,1e-3)
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

![contributors](https://img.shields.io/github/contributors/torchei/torchei)

If you found🧐 any bugs or have🖐️ any suggestions, please tell us.

This repo is open to everyone wants to maintain together.

You can helps us with follow things:

- PR your implemented methods in your or others' papers
- Complete our project
- Translate our docs to your language
- Other

We want to build TorchEI to best toolbox in DNN Reliability for bit flip, adversarial attack, and others.

:e-mail: forcessless@foxmail.com

## Citation

Our paper is under delivering.

## License

> [MIT](https://github.com/TorchEI/TorchEI/blob/main/LICENSE) License.
> Copyright:copyright:2022/5/23-present, Hao Zheng.
