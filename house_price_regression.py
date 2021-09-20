import numpy as np
from numpy.core import numeric
import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv('F:\\train.csv',index_col=0)#读取数据
test=pd.read_csv('F:\\test.csv',index_col=0)#读取数据

# 主要影响因素画图，去离群点
# plt.scatter(x=train.OverallQual, y=train.SalePrice,color='b')
# plt.grid(c='silver',linestyle='--')
# plt.savefig('OverallQual_SalePrice.png')
# plt.show()
train.drop(train[(train['OverallQual']<5)&(train['SalePrice']>200000)].index,inplace=True)

# plt.scatter(x=train.YearBuilt, y=train.SalePrice,color='r')
# plt.grid(c='silver',linestyle='--')
# plt.show()
train.drop(train[(train['YearBuilt']<1900)&(train['SalePrice']>400000)].index,inplace=True)
train.drop(train[(train['YearBuilt']<2000)&(train['SalePrice']>700000)].index,inplace=True)

# plt.scatter(x=train.TotalBsmtSF, y=train.SalePrice,color='g')
# plt.grid(c='silver',linestyle='--')
# plt.show()
train.drop(train[(train['TotalBsmtSF']>6000)&(train['SalePrice']>10000)].index,inplace=True)

# plt.scatter(x=train.GrLivArea, y=train.SalePrice,color='y')
# plt.grid(c='silver',linestyle='--')
# plt.show()
train.drop(train[(train['GrLivArea']>4000)&(train['SalePrice']>100000)].index,inplace=True)

train.reset_index(drop=True, inplace=True)#重置索引

allset=pd.concat([train,test])#去离群点所在行的训练集 合并 测试集

allset=allset.drop_duplicates(keep='first',inplace=False)#去重。没有重复值

na_count=allset.isnull().sum().sort_values(ascending=False)#各列缺失值个数，降序排序
na_ratio=na_count/len(allset)#计算缺失值占比
na_ratio[:36,]

allset.drop('PoolQC',axis=1,inplace=True)
allset.drop('MiscFeature',axis=1,inplace=True)
allset.drop('Alley',axis=1,inplace=True)
allset.drop('Fence',axis=1,inplace=True)
allset.drop('FireplaceQu',axis=1,inplace=True)
allset.drop('LotFrontage',axis=1,inplace=True)

allset.drop('GarageCond',axis=1,inplace=True)
allset.drop('GarageFinish',axis=1,inplace=True)
allset.drop('GarageQual',axis=1,inplace=True)
allset.drop('GarageYrBlt',axis=1,inplace=True)
allset.drop('GarageType',axis=1,inplace=True)

allset.drop('BsmtCond',axis=1,inplace=True)
allset.drop('BsmtExposure',axis=1,inplace=True)
allset.drop('BsmtQual',axis=1,inplace=True)
allset.drop('BsmtFinType2',axis=1,inplace=True)
allset.drop('BsmtFinType1',axis=1,inplace=True)

allset.drop('MasVnrType',axis=1,inplace=True)
allset.drop('MasVnrArea',axis=1,inplace=True)

# 数值型 0填充。
num_cols=["BsmtUnfSF","TotalBsmtSF","BsmtHalfBath","BsmtFullBath","GarageArea","GarageCars",
          "BsmtFinSF1","BsmtFinSF2",]
for col in num_cols:
    allset[col].fillna(0, inplace=True)
# 众数填充
other_cols = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]
for col in other_cols:
    allset[col].fillna(allset[col].mode()[0], inplace=True)
# Utilities
allset = allset.drop(["Utilities"], axis=1)
# Functional
allset["Functional"] = allset["Functional"].fillna("Typ")

na_count=allset.isnull().sum().sort_values(ascending=False)#各列缺失值个数，降序排序
print(na_count)
print(allset)

#房价取对数log1p
# y_vec=np.log1p(np.array(train['SalePrice']))
y_vec=np.array(train['SalePrice'])


# 编码
allset.drop('SalePrice',axis=1,inplace=True)
allset_ob=pd.DataFrame()
allset_num=pd.DataFrame()
for i in allset.columns.values:
    if allset[i].dtypes=='object':
        allset_ob=pd.concat([allset_ob,allset[i]],axis=1)#object型
    else:allset_num=pd.concat([allset_num,allset[i]],axis=1)#数值型
allset_ob=pd.get_dummies(allset_ob)#object型 one hot encode

# feature_num=allset_num.columns.values#数值型
import scipy
from scipy.stats import stats
from scipy.special import boxcox1p
for i in list(allset_num.columns):
    allset_num[i]=boxcox1p(allset_num[i],0)#scipy.stats.boxcox1使满足正态分布,数值有不用boxcox
    # print(scipy.stats.shapiro(allset_num[i])) 
allset_num=pd.DataFrame(allset_num)

# 合并object型和数值型
allset_vec=pd.DataFrame()
allset_vec=pd.concat([allset_ob,allset[i]],axis=1)

# 归一化
for i in list(allset_vec.columns):
    max_a=np.max(allset_vec[i])
    min_a=np.min(allset_vec[i])
    allset_vec[i]=(allset_vec[i]-min_a)/(max_a-min_a)

# max_y=np.max(y_vec)
# min_y=np.min(y_vec)
# for i in range(len(y_vec)):
#     y_vec[i]=(y_vec[i]-min_y)/(max_y-min_y)

# 拆分训练集测试集
train_vec=allset_vec[:len(train)] 
test_vec=allset_vec[len(train):] 

train_vec['1']=1#添加一列1
x=np.matrix(train_vec)
y=np.matrix(y_vec.reshape(y_vec.shape[0],1))#一维转置
xx=x.T*x
xn=np.linalg.pinv(xx)
w=xn*x.T*y

# 回归
test_vec['1']=1
x0=np.matrix(test_vec)
y0=x0*w
print(y0)
y00=pd.DataFrame(y0)
wwords=y00.to_csv("F:\\submission.csv")