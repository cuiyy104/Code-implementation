import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
'''
1）模型太复杂或者太简单，都会让泛化误差高，我们追求的是位于中间的平衡点
2）模型太复杂就会过拟合，模型太简单就会欠拟合
3）对树模型和树的集成模型来说，树的深度越深，枝叶越多，模型越复杂
4）树模型和树的集成模型的目标，都是减少模型复杂度，把模型往图像的左边移动

'''

# 加载乳腺癌数据集
data = load_breast_cancer()
#print(data.feature_names)
#print(data.target_names)
#print(data.data.shape)

#简单建模试一下
rfc=RandomForestClassifier(n_estimators=100,random_state=90)
score_pre=cross_val_score(rfc,data.data,data.target,cv=10).mean()
print("简单建模的准确率为：",score_pre)

#第一步，先调n_estimators参数

scorel=[]

for i in range(0,200,10):
    rfc=RandomForestClassifier(n_estimators=i+1,
                               n_jobs=-1,
                               random_state=90)
    score=cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scorel.append(score)
print(max(scorel),scorel.index(max(scorel))*10+1)
#output:71
plt.figure(figsize=[20,5])
plt.plot(range(1,201,10),scorel)
plt.show()

#第二步，细化学习曲线,已知最好的点在71附近


scorel=[]
for i in range(61,81):
    rfc=RandomForestClassifier(n_estimators=i,
                               n_jobs=-1,
                               random_state=90)
    score=cross_val_score(rfc,data.data,data.target,cv=10).mean()
    scorel.append(score)
print(max(scorel),([*range(61,81)][scorel.index(max(scorel))]))
#output:73
plt.figure(figsize=[20,5])
plt.plot(range(61,81),scorel)
plt.show()
#跑出来是73


#第三步书写网格搜索参数

param_grid = {
    'n_estimators':np.arange(0, 200, 10),
    'max_depth':np.arange(1,20,1),
    'max_leaf_nodes':np.arange(25,50,1),
    'criterion':['gini','entropy'],
    'min_samples_split':np.arange(2,22,1),
    'min_samples_leaf':np.arange(1,11,1),
    'max_features':np.arange(5,30,1)
}


#先调max_depth
'''因为数据集很小，所有在1到20之间试试，数据集如果大就30到50之间试试，可能还需要画出曲线'''
param_grid = {'max_depth':np.arange(1,20,1)}
rfc=RandomForestClassifier(n_estimators=73,
                           random_state=90,
                           n_jobs=-1)
#动态调优函数
GS=GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)
print("调参后的最佳参数：",GS.best_params_)
print("调参后的最佳准确率：",GS.best_score_)
#output:
#调参后的最佳参数： {'max_depth': 8}

#再调整max_features
'''
max_features的默认最小值是sqrt(n_features)，因此我们使用这个值作为调参范围的
最小值。
'''
param_grid = {'max_features':np.arange(5,30,1)}
rfc=RandomForestClassifier(n_estimators=73,
                           max_depth=8,
                           random_state=90,
                           n_jobs=-1)
'''动态调优函数'''
GS=GridSearchCV(rfc,param_grid,cv=10)
GS.fit(data.data,data.target)
print("调参后的最佳参数：",GS.best_params_)
#output: 22

'''之后的几个参数就模仿上面调整就行'''