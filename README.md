
##  House Price
题目网址：https://www.kaggle.com/c/house-prices-advanced-regression-techniques

​        利用房屋信息数据集对房屋价格进行预测。

#### 数据处理

​        本题对训练集中重要元素绘图观察数据分布情况，并对离群点所在行进行删除。

​        将训练集和测试集合并进行数据处理。去除重复值，查看各列缺失值数量及所占比例，对占比进行降序排序，删除缺失值占比过多的列。

​        **数值型数据** ：对数值型数据用0填充。房屋的某项指标的数值缺失，可能表示房屋在该指标的值为0。

​        **非数值型数据** : 对非数值型数据用众数填充。

#### 特征向量

​        在将各特征生成向量时，对数值型数据和非数值型数据有不同处理方法。

​        **数值型数据** ：在数值型数据生成向量时，调用 $scipy$ 内 $boxcox1p$ 使数据满足正态分布，因数据中有0值，因而不使用 $boxcox$ 。

```
import scipy
from scipy.special import boxcox1p
for i in list(allset_num.columns):
    allset_num[i]=boxcox1p(allset_num[i],0)使满足正态分布
```

​        **非数值型数据** : 在非数值型数据生成向量时，调用 $pandas$ 内 $get\_dummies$ 实现 $One-Hot$

编码（独热编码）。

```
allset_ob=pd.get_dummies(allset_ob)
```

​         将两组向量合并，得到特征向量，并针对各列做归一化处理。

#### 模型

​        房屋价格受多重因素影响，可以评估各影响因素对房价的贡献，依次对给出相应信息的房屋预测价格。 本题选用多元线性回归模型，对从上述步骤中提取出的影响因素生成相应向量进行计算。

​        利用最小二乘法对参数进行估计，相关公式如下。

$$\boldsymbol{\hat\omega^*}=({X}^T{X})^{-1}{X}^T{y}\\
f(\hat{{x}}_i)=\hat{{x}}_i^T({X}^T{X})^{-1}{y}$$

​        其中 ${X}$ 为训练集特征向量，最后一列恒为1，$\bold{y}$ 为训练集中的房屋价格，$\hat{\bold{x}}_i$ 为训练集的特征向量，$f(\hat{\bold{x}}_i)$ 为训练结果。


#### 改进方向

​        可以通过对房屋售卖的了解进一步筛选影响因素。对选出的因素进行主成分分析，对相关性较强的因素只保留其中一个。还可以通过优化算法、尝试其他模型进行改进。

####  参考资料
周志华.机器学习[M].北京:清华大学出版社,2016.

[kaggle竞赛项目：房价预测 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/85817133)

[利用线性回归模型进行kaggle房价预测_-CSDN博客](https://blog.csdn.net/weixin_41890393/article/details/83589860)

[Kaggle--房价预测小组报告-CSDN博客](https://blog.csdn.net/D_i_k_y/article/details/80954961) 

