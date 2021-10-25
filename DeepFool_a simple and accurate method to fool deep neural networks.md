## DeepFool: a simple and accurate method to fool deep neural networks

年份：2016

### 1. Introduction

- 作者首先定义了模型$\hat{k}$对于某一个样本$x$的鲁棒性$\Delta(x;\hat{k})$：
  $$
  \Delta(x;\hat{k})=\min_r ||r||_2\ {\rm subject\ to\ }\hat{k}(x+r)\neq \hat{k}(x)
  $$
  则模型整体的鲁棒性就可以定义为：
  $$
  \rho_{adv}(\hat{k})=\mathbb{E}_x \frac{\Delta(x;\hat{k})}{||x||_2}
  $$
  其中$\mathbb{E}_x$是在数据分布上的期望。

- 作者初步介绍本文有三点贡献：

  - 提出一种简单但是有效的计算和比较模型鲁棒性的方法；
  - 通过大量的实验比较证明了1）本文的方法能有效的计算和比较模型鲁棒性；2）利用对抗样本进行学习能大大提升模型的鲁棒性；
  - 证明如果用于计算对抗样本扰动的方法不准确，那么会对模型鲁棒性的某些结论造成很大的影响。本文提出的方法提供了一个更好了解该现象及其影响的角度；

