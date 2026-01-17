from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing as fch #加利福尼亚房屋价值数据集
import pandas as pd

housevalue = fch()
X = pd.DataFrame(housevalue.data) #放入DataFrame中便于查看
y = housevalue.target
print(X.shape)
print(y.shape)
print(X.head())
print(y[:5])
print(housevalue.feature_names)
X.columns = housevalue.feature_names

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.3,random_state=420)
for i in [Xtrain,Xtest]:
    i.index=range(i.shape[0])

print(Xtrain.shape)

reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtest)
print(yhat.shape)
print(reg.coef_)
print([*zip(Xtrain.columns,reg.coef_)])
print(reg.intercept_)