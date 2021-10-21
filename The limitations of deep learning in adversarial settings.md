## The limitations of deep learning in adversarial settings

年份：2016

### 1 Introduction

- 展示了当前在手写数字识别上对抗样本是如何欺骗DNN的。<img src="Pics/屏幕截图 2021-10-14 105513.png"  />

- 目前生成对抗样本的方法大多是利用梯度网络训练时的梯度，去更新原始的input，而不是更新网络的权重。
- 本文的工作主要有三个：
  - 本文形式化表达了针对分类DNN的对抗器的空间（space of adversaries），描述了对抗器的目标和能力。通过这一形式化表达，我们能够更好地理解对抗器的能力是如何影响对抗器的策略和效果。
  - 本文提出了一种新的只使用DNN结构信息就能够生成对抗样本的算法，不再需要DNN训练时所获得的梯度。该算法首先获得forward derivatives，用以推断DNN模型的学习行为，然后构造adversarial saliency maps去高效地寻找对抗样本。
  - 本文使用计算机视觉领域的DNN来证明上述对抗样本生成算法的有效性，并探究如何防御这些对抗样本的方法。

### 2 Taxonomy of Threat Models in Deep Learning

这一部分主要是给深度学习中的攻击模型进行分类（分类图如下所示），并介绍了几个有关攻击模型能力的前任工作（previous works with respect to the strength of the modeled adversary）。![](Pics/屏幕截图 2021-10-14 162627.png)

#### A. About Deep Neural Networks

这一部分主要回顾了DNN的主要结构，并指明本文主要针对的是监督学习下的多分类器学习问题。

#### B. Adversarial Goals

攻击器的目标可以分为以下四类：

- Confidence reduction：降低原模型对输出的置信度（confidence）；
- Misclassification：使原模型无法正确分类为对应的类别；
- Targeted misclassification：生成一些inputs，并让原模型将这些样本错分为某一特定的类别；
- Source/target misclassification：给原始样本添加一些微小的扰动，让原模型将这些样本错分为某一特定的类别；

#### C. Adversarial Capabilities

攻击器之间有不同的能力区别，这很大程度上取决于攻击器掌握的信息。本文所研究的都是测试阶段的攻击，不考虑训练阶段的攻击。不同的攻击器大致有以下信息了解程度（能力强弱程度），按照能力由强到弱进行排列：

- Training data and network architecture：拥有perfect knowledge，知道训练数据以及训练好的模型的所有结构，以及用于训练该模型的方法。这是最强大的模型；
- Network architecture：知道模型的所有信息，包括结构和各个参数的值。因此攻击器拥有足够的信息来模拟这个网络（simulate the network）；
- Training data：知道训练数据集的分布。因此攻击器可以按照该分布收集一个surrogate dataset，然后利用常用的DNN结构在该数据集上训练，用以模拟真实原模型；
- Oracle：攻击器能够使用该神经网络。因此攻击器可以使用differential attacks，通过观察提供的输入和模型的输出之间的关系或变化（即可以不断地改变input，并观察对应的output）来构造对抗样本；
- Samples：攻击器只能收集到相关的输入输出对，却不能够自行修改input来观察模型output的变化。这是最难攻击的类型，也是能力最弱的类型；

### 3 Approach

这一部分主要介绍本文提出的构造对抗样本的方法。该方法只需要知道模型的结构和训练后的参数，就可以利用forward derivative和adversarial saliency map构造出对抗样本（即图2中星号所在位置 ）。

#### A. Studying a Simple Neural Network

该部分利用简单的三层感知机（只有一层隐藏层）做演示，介绍如何使用forward derivative构造对抗样本。

让模型学习“and”函数${\rm \mathbf{F}}$，输入为$X=(x_1,x_2),X\in [0,1]^2$，输出为$Y$，且小数将被四舍五入为整数。模型训练完后结果可视化为下图：![](Pics/屏幕截图 2021-10-16 104911.png)

从图中其实可以看出该模型学习的知识大体上是正确的。

forward derivative的定义即为函数${\rm \mathbf{F}}$的雅各比矩阵（Jacobian matrix）。在这里由于${\rm \mathbf{F}}$的输出值是一维的，所以该雅各比矩阵被定义为：
$$
\nabla{\rm \mathbf{F}}(X)=[\frac{\partial{\rm \mathbf{F}}(X)}{\partial x_1},\frac{\partial{\rm \mathbf{F}}(X)}{\partial x_2}]
$$
下图即为对于不同的输入$(x_1,x_2)$，$\frac{\partial{\rm \mathbf{F}}(X)}{\partial x_2}$的梯度变化图（因为$x_1$和$x_2$对称，所以只用研究一个）：![](Pics/屏幕截图 2021-10-16 105842.png)

从图中可以注意到，在梯度突变处函数的输出结果是变化剧烈的，而在其他梯度平稳处输出结果则基本不变。因此，这就提示我们应该在forward derivative值比较大的区域寻找对抗样本。

通过这一实验，我们可以得出以下三个结论：

- 输入的微小扰动可以给输出带来剧烈的变化；
- 不是输入空间中的所有区域都能简单地找到对抗样本，只有梯度变化较剧烈的部分可以；
- 利用forward derivative方法可以缩小对抗样本的寻找空间；

#### B. Generalizing to Feedforward Deep Neural Networks

该部分作者将A中的方法泛化至任何DNN，只要是一个非循环的、且激活函数可导的DNN即可。具体的字母标识如下图所示。![](Pics/屏幕截图 2021-10-19 110518.png)

构造对抗样本的算法则如下所示：![](Pics/屏幕截图 2021-10-19 110600.png)

该算法主要有三个基本步骤：

- 计算Forward Derivative（前向导数）。如A中所示，前向导数其实就是所学函数的雅各比矩阵，并且其计算的梯度和BP反传十分类似，不过有两个明显不同。第一，前向导数是直接利用网络的输出求导，而BP反传是利用损失函数求导；第二，前向导数是对输入直接求偏导，BP反传则是对模型的参数求偏导。前向导数的意义就是试图寻找输入中的哪些分量会对模型的输出产生较大的影响。

  我们现在考虑一个$(i,j)\in [1..M]\times [1..N]$对，$i$代表输入的第$i$个分量，$j$代表输出的第$j$个分量。那么每个隐藏层对$x_i$求导可得：
  $$
  \begin{aligned} \frac {\partial \mathbf{H}_k(X)}{\partial x_i} &=\left[\frac{\partial f_{k,p}(\mathbf{W}_{k,p} \cdot \mathbf{H}_{k-1}+b_{k,p})}{\partial x_i}\right]_{p\in 1..m_k} \\ & = \left(\mathbf{W}_{k,p} \cdot \frac{\partial \mathbf{H}_{k-1}}{\partial x_i} \right) \times \frac{\partial f_{k,p}}{\partial x_i}(\mathbf{W}_{k,p} \cdot \mathbf{H}_{k-1}+b_{k,p})  \end{aligned}
  $$
  其中$\mathbf{H}_k$是第$k$个隐藏层的输出向量，$f_{k,p}$是第k层第$p$个神经元的激活函数。依此类推直到最后的输出层。

  根据我们之前定义的攻击器能力，我们知道上述式子所有的参数，除了$\frac{\partial \mathbf{H}_n}{\partial x_i}$，而这个值需要从输入层开始一层层地向前计算（这也是前向导数得名由来），直到最终的输出层。

- 计算Adversarial Saliency Maps（下简称ASM）。ASM是一种problem-specific的对抗攻击方法，根据不同的任务会有不同的定义形式。下面将以分类问题作为例子介绍。

  在分类问题中，攻击器想让模型将样本$X$错分为类别$t\neq label(X)$。为了达到这一目的，模型$\mathbf{F}$输出目标类别$t$的概率就应该增加，而输出其他类别$j\neq t$的概率就应该下降，直到$t=\mathop{\rm arg\  max}\limits_{j}\mathbf{F}_j(X)$。

  为了达到这一目的，我们可以对输入计算ASM，即$S(X,t)$：
  $$
  S(X,t)[i]=\begin{cases} 0\  {\rm if}\ \frac{\partial \mathbf{F}_t(X)}{\partial X_i}<0\ {\rm or}\ \sum_{j\neq t}\frac{\partial \mathbf{F}_j(X)}{\partial X_i}>0 \\ \left(\frac{\partial \mathbf{F}_t(X)}{\partial X_i}\right)\left|\sum_{j\neq t}\frac{\partial \mathbf{F}_j(X)}{\partial X_i}\right|\ {\rm otherwise} \end{cases}
  $$
  由上述公式我们易得，ASM中取值较高的分量对应的就是输入分量中既能提升target class概率又能降低其他classes概率的分量。因此攻击器增加这些输入分量的值，最终就可以让原模型将样本错分为target class。将ASM可视化出来即为：![](Pics/屏幕截图 2021-10-21 113023.png)

  当然我们也可以利用前向导数构造其他的ASM，比如找到输入中应该被降低的分量有哪些。

- 对样本进行扰动。当我们利用ASM确定了需要扰动的输入分量后，就可以对其进行扰动。需要保证扰动不能超过我们所设置的maximum distortion，即算法中的$\Upsilon$。

### 4 Application of the Approach

