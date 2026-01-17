from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing as fch #加利福尼亚房屋价值数据集
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
import sklearn
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


housevalue = fch()
X = pd.DataFrame(housevalue.data) #放入DataFrame中便于查看
y = housevalue.target

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
for i in [Xtrain,Xtest]:
    i.index=range(i.shape[0])
reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtest)

print(MSE(Ytest, yhat))
print(y.max(), y.min())
#cross_val_score(reg,X,y,cv=10,scoring="mean_squared_error")

score_= cross_val_score(reg,X,y,cv=10,scoring="neg_mean_squared_error")
print(score_)
'''注意cross_val_score默认是越大越好，所以对于MSE这种越小越好的指标，需要加个负号变成负的均方误差'''

r2=r2_score(y_true = Ytest,y_pred = yhat)
print(r2)
'''R²也叫决定系数，反映自变量对因变量的解释程度，取值范围为0-1，越接近1说明模型越好'''
'''如果R²为负数，说明模型很差，甚至不如用因变量的均值来预测效果好'''
print(cross_val_score(reg,X,y,cv=10,scoring="r2").mean())

plt.plot(range(len(Ytest)),sorted(Ytest),c="black",label= "Data")
plt.plot(range(len(yhat)),sorted(yhat),c="red",label = "Predict")
plt.legend()
plt.show()
'''从图中可以看出，预测值与真实值还是有一定差距的'''