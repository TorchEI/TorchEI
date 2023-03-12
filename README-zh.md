<div align="center">
  <img src="https://raw.githubusercontent.com/TorchEI/TorchEI/main/assets/torchei.svg" alt="torchei_logo" align="center" style="width:30%;"  />
</div>

<h1 style = "margin:0;" align="center">TorchEI⚡</h1>

<div align = "center" style="font-weight: bold;"><a href="#introduction">Intro</a> ● <a href="#quick-example">Usage</a> ● <a href="https://TorchEI.github.com.io/TorchEI/">Doc</a>  ● <a href="#citation">Cite</a> ● <a href="#contribution" >Contribution</a> ● <a href="#license">License</a></div>

------

<div align = "center">
    <a href = "https://github.com/TorchEI/TorchEI/actions/workflows/pytest-cov.yml">
  <img src="https://github.com/TorchEI/TorchEI/actions/workflows/pytest-cov.yml/badge.svg"/></a>
    <a href = "https://github.com/TorchEI/TorchEI/actions/workflows/doc-deploy.yml">
  <img src="https://github.com/TorchEI/TorchEI/actions/workflows/doc-deploy.yml/badge.svg"/></a>
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

👋TorchEI, 发音为*/ˈtôrCHər/*(like torture),  是Pytorch Error Injection的缩写, 一个围绕DNN Reliability 研究的高速工具箱. TorchEI 使您能够快速简单地将错误注入 DNN，收集您需要的信息并强化您的 DNN。


## Features

- 完善的类型提示和文档支持
- 包含来自 DNN 可靠性论文的方法
- 高度定制化

## Quick Example

在这里，我们将向您展示一个简单的示例，或者您可以尝试 [interactive demo](https://colab.research.google.com/github/TorchEI/TorchEI/blob/main/example.ipynb) 和[online editor](https://github.dev/TorchEI/TorchEI)

#### Installing

你可以使用  `pip3 install torchei` 安装或 [下载](https://github.com/TorchEI/TorchEI/archive/refs/heads/main.zip) 

#### Example

初始化故障模型

```python
import torch
from torchvision import models
import torchei
model = models.resnet18(pretrained=True)
data = torch.load('data/ilsvrc_valid8.pt')
fault_model = torchei.fault_model(model,data)
```

使用emat方法计算可靠性

```python
fault_model.emat_attack(10,1e-3)
```

使用[SERN](https://dl.acm.org/doi/abs/10.1145/3386263.3406938)方法计算可靠性 

```python
fault_model.sern_calc(output_class=1000)
```

使用ODR方法加固DNN

```python
fault_model.outlierDR_protection()
fault_model.emat_attack(10,1e-3)
```

## Contribution

 ![contributors](https://img.shields.io/github/contributors/torchei/torchei)

如果您发现🧐任何错误或有🖐️任何建议，请告诉我们。

这个 repo 欢迎所有想要一起维护的人。

You can helps us with follow things:

- PR your implemented methods in your or others' papers
- Complete our project
- Translate our docs to your language
- Other

我们希望将 TorchEI 构建为 DNN 可靠性方面的最佳工具箱，用于位翻转、对抗性攻击等。

:e-mail: forcessless@foxmail.com

## Citation

Our paper is under delivering.

## License

> [MIT](https://github.com/TorchEI/TorchEI/blob/main/LICENSE) License.
> Copyright:copyright:2022/5/23-present, Hao Zheng.
