##Intriguing properties of neural networks
年份：2014

### 1 Introduction

- 人工神经网络因其可以随意表达大量的并行非线性计算而性能优越。但是由于其结果是通过自动的bp反传获得的结果，所以很难解释，并且会产生违反直觉的特性。本文主要讨论深度神经网络的两个反直觉特性。
- 第一个特性：神经单元的语义信息。本文的结果显示是激活的高维空间（entire space of activation）包含了大量的语义信息，而不是单个神经单元包含语义信息。
- 第二个特性：对一个测试图片加上一个非随机的几乎不可察觉的扰动，训练好的物体识别神经网络分类器就会对其的判断出错。这些扰动后的样本就称为adversarial examples。同时这些扰动是robust的，对不同结构、在不同训练集上训练的模型都能产生攻击效果。这说明神经网络有其内在的blind spots，且它的结构和数据的分布有很大的不明显（non-obvious）的关联。

### 2 Framework

- 记号表示：$x \in R$表示输入的图像。$\phi(x)$代表某些层的激活函数的输出值。
- 对于不同的数据集才用了不同的模型进行测试
    - MINST数据集用了全联接网络（FC）和Autoencoder（AE）；
    - ImageNet数据集用了AlexNet；
    - Youtube上收集的图片用了QuocNet（一种非监督学习的模型）；

### 3 Unit of: $\phi(x)$

- 之前的研究希望将隐藏层单元的激活函数输出值作为一个有意义的特征，因此它们会寻找能让该特征最大化的图片，借以研究它们的特点。即
    $$
    x'=argmax_{x \in I}\langle \phi(x),e_i \rangle
    $$
    其中$I$为模型没有接触过的图片数据，$e_i$是与第$i$个隐藏层单元相关的自然基向量。 

- 但是本文的研究发现，任意的一个向量$v \in R^n$也能达到类似自然基向量的作用，即$x'$图片也具有语义上的相关性：
    $$
    x'=argmax_{x \in I} \langle \phi(x),v \rangle
    $$
    这样的结果说明，在探究$\phi(x)$的特性时，自然基向量并没有比随机向量更好解释它，因此神经网络可能不是通过坐标（cooridinates）来分解不同因素的作用。

###4 Blind Spots in Neural Networks

- 当前的许多研究都认为，在输入和输出之间堆积更多的非线性层是为了获得input space上的non-local generalization prior，即如果训练数据在input space的某个区域内没有出现过，那么输出层就会对这个区域的信息给予很小的关注度，也就是会得到很小的概率值。
- 这样的结果就表示，local generalization应该是work的，即对于足够小的$\epsilon>0$，一个训练集中的图片$x$，$x+r$如果满足条件$||r||<\epsilon$，那么$x+r$有很大的概率模型会将其分到正确的类别当中。这种特性也可以被称为“平滑性”。
- 但是本文发现，这种平滑性是不成立的，给予原图一个非常微小的改变，就可以让模型对其的分类产生非常大的偏差。
- 本文也提到他们所提出的优化方法和hard-negative mining方法有异曲同工之妙。hard-negative mining是指找到那些模型本应被给予高概率值但是却给予低概率值的样本，利用他们调整训练数据集的数据分布，再次训练模型。

#### 4.1 Formal description

首先介绍基本符号，$f:R^m \rightarrow\{1... k\}$是一个将图片分类的分类器，并且其有一个连续的损失函数$loss_f:R^m \times \{1...k\} \rightarrow R^+$​。$l$为目标类别。我们的优化目标即为
$$
Minimize\ ||r||_2 \ subject\ to: \\
f(x+r)=l \\
x+r \in [0,1]^m
$$
由于直接进行求解非常困难，因此作者采用了box-constrained L-BFGS算法来近似。总结说来，就是进行如下的优化问题：
$$
Minimize\ c|r|+loss_f(x+r,l)\ subject\ to:\\
f(x) \neq l\\
f(x+r)=l\\
x+r \in [0,1]^m
$$

#### 4.2 Experimental results
