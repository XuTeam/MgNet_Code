# MgNet: A Unified Framework of Multigrid and Convolutional Neural Network
This repository contains the PyTorch implementation of MgNet. 

As an example, the following command trains a MgNet with depth L=100 and growth rate k=12 on CIFAR-10:



### Introduction

For simplicity, we use the following notation to represent different MgNet models with different 
hyper-parameters:

~~~latex
```math
SE = \frac{\sigma}{\sqrt{n}}
```
~~~



​	{\rm MgNet}[\nu_1,\cdots,\nu_J]\text{-}[(c_{u,1}, c_{f,1}), \cdots, (c_{u,J}, c_{f,J})]\text{-}B^{\ell,i}.
These hyper-parameters are defined as follows.
\begin{itemize}
​	\item $[\nu_1,\cdots,\nu_J]$: The number of smoothing iterations on each grid. For example, $[2,2,2,2]$ means that there are 4 grids, and the number of iterations of each grid is 2.
​	%ResNet18 since it shares almost the same coding structure if you consider smoothers as basic blocks in ResNet models.
​	\item $[(c_{u,1}, c_{f,1}), \cdots, (c_{u,J}, c_{f,J})]$: The number of channels for $u^{\ell,i}$ and $f^\ell$ on each grid. We mainly consider the case $c_{u,\ell} = c_{f,\ell}$, which suggests us the following simplification notation $[c_{1}, \cdots, c_{J}]$, or even $[c]$ if we further take $c_{1}=c_2=\cdots=c_{J}$. For examples, 
​		${\rm MgNet}[2,2,2,2]\text{-}[64,128,256,512]$ and ${\rm MgNet}[2,2,2,2]\text{-}[256]$.
​	\item $B^{\ell,i}$: This means that we use different smoother $B^{\ell,i}$ in each smoothing iteration. Correspondingly, $B^{\ell}$ means that we share the smoother among each grid, which is
​	\begin{equation}\label{eq:Bl}
​		u^{\ell,i} = u^{\ell,i-1} + \sigma \circ B^{\ell} \ast \sigma\left({f^\ell -  A^{\ell} \ast u^{\ell,i-1}}\right).
​	\end{equation}
​	Here we mention that we always use $A^{\ell}$, which only depends on grids.
\end{itemize}
For example, the following notation ${\rm MgNet}[2,2,2,2]\text{-}[256]\text{-}B^{\ell},$
denotes a MgNet model which adopts 4 different grids (feature resolutions), 2 smoothing iterations on each grid, 256 channels for both feature tensor $u^{\ell,i}$ and data tensor $f^\ell$, and \eqref{eq:Bl} as the smoothing iteration. 



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

