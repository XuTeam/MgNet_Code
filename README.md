# MgNet: A Unified Framework of Multigrid and Convolutional Neural Network
This repository contains the PyTorch implementation of MgNet. 

As an example, the following command trains a MgNet with  on CIFAR-10:

`python ******************************(Jianqing, please add this.)`

### Introduction

MgNet is a unified model that simultaneously recovers some convolutional neural networks (CNN) for image classification and multigrid (MG) methods for solving discretized partial differential equations (PDEs). Here is a diagram of its architecture.

![MgNet](./figures/MgNet.png)

For simplicity, we use the following notation to represent different MgNet models with different hyper-parameters: 

${\rm MgNet}[\nu_1,\cdots,\nu_J]\text{-}[(c_{u,1}, c_{f,1}), \cdots, (c_{u,J}, c_{f,J})]\text{-}B^{\ell,i}.$ 

These hyper-parameters are defined as follows. 

1. $[\nu_1,\cdots,\nu_J]$: The number of smoothing iterations on each grid. For example, $[2,2,2,2]$ means that there are 4 grids, and the number of iterations of each grid is 2.
2. $[(c_{u,1}, c_{f,1}), \cdots, (c_{u,J}, c_{f,J})]$: The number of channels for $u^{\ell,i}$ and $f^\ell$ on each grid. We mainly consider the case $c_{u,\ell} = c_{f,\ell}$, which suggests us the following simplification notation $[c_{1}, \cdots, c_{J}]$, or even $[c]$ if we further take $c_{1}=c_2=\cdots=c_{J}$. For examples, ${\rm MgNet}[2,2,2,2]\text{-}[64,128,256,512]$ and ${\rm MgNet}[2,2,2,2]\text{-}[256]$.
3. $B^{\ell,i}$: This means that we use different smoother $B^{\ell,i}$ in each smoothing iteration. Correspondingly, $B^{\ell}$ means that we share the smoother among each grid, which is $u^{\ell,i} = u^{\ell,i-1} + \sigma \circ B^{\ell} \ast \sigma\left({f^\ell -  A^{\ell} \ast u^{\ell,i-1}}\right).$ 

Here we mention that we always use $A^{\ell}$, which only depends on grids. For example, the following notation ${\rm MgNet}[2,2,2,2]\text{-}[256]\text{-}B^{\ell},$ denotes a MgNet model which adopts 4 different grids (feature resolutions), 2 smoothing iterations on each grid, 256 channels for both feature tensor $u^{\ell,i}$ and data tensor $f^\ell$ as the smoothing iteration. 

### Citation

If you find MgNet useful in your research, please consider citing:

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

