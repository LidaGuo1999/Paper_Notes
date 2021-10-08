## EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES

年份：2015

### 1 Introduction

- 本文作者首先回顾了论文”Intriguing properties of Neural Networks“里面的结论：对抗样本的存在实际上揭示了我们训练算法中的基本盲点（blind spots）；
- 以前的研究人文对抗样本主要是由于深度神经网络的极端非线性（extreme nonlinearity），同时还有模型训练的不平衡和不完全的正则化（regularization）。本文的作者却证明高维空间中的线性操作已经足够造成对抗样本的存在；
- 普通的正则化方法（如dropout，pre-training，model averaging）不太能显著提高模型抵御对抗样本的能力，但是采用非线性的模型却可以做到；
- 当下，线性模型的易训练性和其难以抵御对抗攻击的不安全性产生了矛盾。也许在未来，会有更有效的优化方法的出现，使得对于非线性模型的训练更加高效；

### 2 Related Work

- 前人所作的研究说明，即使是在测试集上表现得很好的模型，其实也没有真正学到正确的知识和概念。它们的能力是虚假的，仅仅能够在数据分布中出现较多次数的样本上表现优越。这样的结果其实是很disappointing的，因为目前很多CV中的研究都是利用卷积神经网络构造空间，再在空间中使用欧式距离进行相关应用的。对抗样本的出现说明即使距离很近，仍然会导致模型的错误判别。
- 目前有许多学者已经开始着手设计能够抵御对抗攻击的模型，但是还没能够做到既保证模型对非攻击样本的高正确性，同时也抵御攻击样本的攻击。

### 3 The Linear Explanation of Adversarial Examples

- 在许多的实际问题当中，输入的维度是有限的，比如图片的每个像素点用8位来表示，意味着他们将1/255以下的动态信息都舍弃了。也正因为如此，如果我们对原始输入$x$添加一个小于输入精度的扰动$\eta$，得到一个新输入$\tilde{x}=x+\eta$，那么分类器对这个新输入的分类应该与原结果没有区别。如果我们能满足$||\eta||_\infty < \epsilon$，其中$\epsilon$是一个小于输入精度的值，那么分类器应该将$x$和$\tilde{x}$分为同类。

- 让我们考虑权重向量$w$和对抗样本$\tilde{x}$之间的点积运算：
  $$
  w^\top \tilde{x}=w^\top x+w^\top \eta
  $$
  增加了扰动之后，该层的输出增加了$w^\top \eta$。如果我们假设$w$有的维度为$n$，并且每个维度的平均值为$m$，且$\eta =sign(w)$（即$\eta$的每个维度符号都和$w$相同），那么最终该层的输出将会最多增长$\epsilon mn$。
  
- 因此本文的作者认为线性就已经可以解释对抗样本所出现的原因了，当输入的维度足够大时，一个简单的线性模型就完全能够被对抗样本所攻击。

### 4 Linear Perturbation of Non-Linear Models

- 由于对抗样本可以与模型的线性有关，因此可以使用线性的方法方便地生成对抗样本。作者认为许多神经网络其实都是线性网络，如LSTMs，ReLUs和maxout网络等等（Sigmod网络属于非线性）。

- 作者所提出的构建对抗样本的方法可以被称作“fast gradient sign methos”。具体来讲，我们令$\theta$为模型的参数，$x$为模型的输入，$y$为模型训练的真实标签（对于非监督学习也可以没有），$J(\theta,x,y)$​为损失函数。则所添加给输入的扰动即为：
  $$
  \eta=\epsilon sign(\nabla_xJ(\theta,x,y))
  $$
  所需要的梯度根据BP反传就可以快速地计算。

- 实验的结果也非常喜人，该方法可以快速地构造对抗样本，并且让模型对错误分类给予很高的置信度。![](Pics/屏幕截图 2021-10-08 171141.png)

- 作者认为这样的实验结果从一个侧面证明了对抗样本能够攻击成功的原因是由于线性（linearity）。作者在实验中也发现将输入$x$朝着梯度下降的方向旋转一个很小的角度也可以产生对抗样本。

### 5 Adversarial Training of Linear Models Versus Weight Decay

- 
