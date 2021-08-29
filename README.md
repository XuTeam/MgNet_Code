# MgNet: A Unified Framework of Multigrid and Convolutional Neural Network
This repository contains the [PyTorch](https://pytorch.org/) (1.7.1) implementation of MgNet. 

As an example, the following command trains a MgNet with  on CIFAR-100:

`python mgnet_train.py --wise-B --dataset cifar100`

### Introduction

MgNet is a unified model that simultaneously recovers some convolutional neural networks (CNN) for image classification and multigrid (MG) methods for solving discretized partial differential equations (PDEs). Here is a diagram of its architecture.

![MgNet](./figures/MgNet.png)

For simplicity, we use the following notation to represent different MgNet models with different hyper-parameters: 

<a href="https://www.codecogs.com/eqnedit.php?latex={\rm&space;MgNet}[\nu_1,\cdots,\nu_J]\text{-}[(c_{u,1},&space;c_{f,1}),&space;\cdots,&space;(c_{u,J},&space;c_{f,J})]\text{-}B^{\ell,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\rm&space;MgNet}[\nu_1,\cdots,\nu_J]\text{-}[(c_{u,1},&space;c_{f,1}),&space;\cdots,&space;(c_{u,J},&space;c_{f,J})]\text{-}B^{\ell,i}" title="{\rm MgNet}[\nu_1,\cdots,\nu_J]\text{-}[(c_{u,1}, c_{f,1}), \cdots, (c_{u,J}, c_{f,J})]\text{-}B^{\ell,i}" /></a>

These hyper-parameters are defined as follows. 

1. <img src="https://latex.codecogs.com/gif.latex?[\nu_1,\cdots,\nu_J]" title="[\nu_1,\cdots,\nu_J]" />: The number of smoothing iterations on each grid. For example, [2,2,2,2] means that there are 4 grids, and the number of iterations of each grid is 2.
2. <img src="https://latex.codecogs.com/gif.latex?[(c_{u,1},&space;c_{f,1}),&space;\cdots,&space;(c_{u,J},&space;c_{f,J})]" title="[(c_{u,1}, c_{f,1}), \cdots, (c_{u,J}, c_{f,J})]" />: The number of channels for <img src="https://latex.codecogs.com/gif.latex?u^{\ell,i}" title="u^{\ell,i}" /> and <img src="https://latex.codecogs.com/gif.latex?f^{\ell}" title="f^{\ell}" /> on each grid. We mainly consider the case <img src="https://latex.codecogs.com/gif.latex?c_{u,\ell}&space;=&space;c_{f,\ell}" title="c_{u,\ell} = c_{f,\ell}" />, which suggests us the following simplification notation <img src="https://latex.codecogs.com/gif.latex?[c_{1},&space;\cdots,&space;c_{J}]" title="[c_{1}, \cdots, c_{J}]" />, or even [c] if we further take <img src="https://latex.codecogs.com/gif.latex?c_{1}=c_2=\cdots=c_{J}" title="c_{1}=c_2=\cdots=c_{J}" />. For examples, <img src="https://latex.codecogs.com/gif.latex?{\rm&space;MgNet}[2,2,2,2]\text{-}[64,128,256,512]" title="{\rm MgNet}[2,2,2,2]\text{-}[64,128,256,512]" /> and <img src="https://latex.codecogs.com/gif.latex?{\rm&space;MgNet}[2,2,2,2]\text{-}[256]" title="{\rm MgNet}[2,2,2,2]\text{-}[256]" />.
3. <img src="https://latex.codecogs.com/gif.latex?B^{\ell,i}" title="B^{\ell,i}" />: This means that we use different smoother <img src="https://latex.codecogs.com/gif.latex?B^{\ell,i}" title="B^{\ell,i}" /> in each smoothing iteration. Correspondingly, <img src="https://latex.codecogs.com/gif.latex?B^{\ell}" title="B^{\ell}" /> means that we share the smoother among each grid, which is <img src="https://latex.codecogs.com/gif.latex?u^{\ell,i}&space;=&space;u^{\ell,i-1}&space;&plus;&space;\sigma&space;\circ&space;B^{\ell}&space;\ast&space;\sigma\left({f^\ell&space;-&space;A^{\ell}&space;\ast&space;u^{\ell,i-1}}\right)." title="u^{\ell,i} = u^{\ell,i-1} + \sigma \circ B^{\ell} \ast \sigma\left({f^\ell - A^{\ell} \ast u^{\ell,i-1}}\right)." /> 

Here we mention that we always use <img src="https://latex.codecogs.com/gif.latex?A^{\ell}" title="A^{\ell}" />, which only depends on grids. For example, the following notation <img src="https://latex.codecogs.com/gif.latex?{\rm&space;MgNet}[2,2,2,2]\text{-}[256]\text{-}B^{\ell}" title="{\rm MgNet}[2,2,2,2]\text{-}[256]\text{-}B^{\ell}" /> denotes a MgNet model which adopts 4 different grids (feature resolutions), 2 smoothing iterations on each grid, 256 channels for both feature tensor <img src="https://latex.codecogs.com/gif.latex?u^{\ell,i}" title="u^{\ell,i}" /> and data tensor <img src="https://latex.codecogs.com/gif.latex?f^{\ell}" title="f^{\ell}" />, and smoothing iteration <img src="https://latex.codecogs.com/gif.latex?B^{\ell}" title="B^{\ell}" />. 

### Results on CIFAR and ImageNet

#### **Jianqing, please add Table 4 in linear model paper here.** 

Model                    | Parameters | CIFAR-10 | CIFAR100
-----                       | -----           | -----        |  ------
MgNet[2,2,2,2],256,B^{l} |  8.3M          | 96.00       | 79.94





### Citation

For more detials about MgNet, we refer to the following two papers. If you also find MgNet useful in your research, please consider citing:

```
@article{he2019mgnet,
  title={MgNet: A unified framework of multigrid and convolutional neural network},
  author={He, Juncai and Xu, Jinchao},
  journal={Science china mathematics},
  volume={62},
  number={7},
  pages={1331--1354},
  year={2019},
  publisher={Springer}
}

@article{he2019constrained,
  title={Constrained Linear Data-feature Mapping for Image Classification},
  author={He, Juncai and Chen, Yuyan and Zhang, Lian and Xu, Jinchao},
  journal={arXiv preprint arXiv:1911.10428},
  year={2019}
}
```

### Contact

jhe AT utexas.edu

xu AT math.psu.edu

zhanglian AT multigrid.org

jqzhu AT emails.bjut.edu.cn

Any discussions, comments, suggestions and questions are welcome!

