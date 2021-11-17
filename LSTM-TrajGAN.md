## LSTM-TrajGAN: A Deep Learning Approach to Trajectory Privacy Protection

会议：GIScience

年份：2020

### 1 Introduction

在这一部分，作者主要总结了对应的任务和当前的发展情况。

- 轨迹隐私（trajectory privacy）指的是个人保护自身轨迹以及某些常去地点不被泄露的权力，但如今大数据手段让轨迹隐私难以得到应有的保护；
- 最简单的办法是remove the identifiers，比如名字、ID等等。但是凭借其他信息，机器学习模型仍旧能推断出轨迹的拥有者相关信息；还有一种是将轨迹点都aggregate起来，但是这样不仅难以保护隐私，还会使得相关的时空任务效果下降；
- 作者把当前的隐私保护方法分为以下两类：
  - 将不同用户的不同轨迹给组合或者混合起来（grouping and mixing），形成一个k匿名模型。
  - 在原始轨迹数据上添加一定的扰动，即geomasking。
- 虽然上述方法在一定程度上可以保护隐私，但是也有几个明显的问题：
  - 上述方法的本质其实可以概括为在原来的轨迹信息上添加扰动，但是这也会影响我们利用它们进行其他任务的精度，因此这其中的trade-off很难平衡；
  - 上述方法主要使用的是空间信息，较少使用时序信息或者其他信息；
  - 上述方法比较依赖人工定义的procedures；

