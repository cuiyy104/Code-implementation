from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier as KNN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv(r'digit recognizor.csv')
#print(data.shape)
#print(data.head())
X=data.iloc[:,1:]
y=data.iloc[:,0]
#画出方差累计曲线
pca_line=PCA().fit(X)
plt.figure(figsize=[20,5])
plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
plt.show()

#用不同的n_components评估随机森林分类器的性能
score=[]
for i in range(11,51,10):
    X_dr=PCA(i).fit(X).transform(X)
    once=cross_val_score(RFC(n_estimators=10,random_state=0)
                         ,X_dr,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=[20,5])
plt.plot(range(11,51,10),score)
plt.show()
print(int(np.argmax(score))*10+11)
#output:21
'''细化曲线，不跑了电脑太慢，这里感觉是23'''

X_dr=PCA(23).fit(X).transform(X)
score_PCA=cross_val_score(RFC(n_estimators=10,random_state=0),X_dr,y,cv=5).mean()
print(f"PCA降维后随机森林得分:{score_PCA}")
#这里换成KNN模型

score = []
for i in range(10):
   X_dr = PCA(23).fit_transform(X)
   once = cross_val_score(KNN(i+1),X_dr,y,cv=5).mean()
   score.append(once)
plt.figure(figsize=[20,5])
plt.plot(range(10),score)
plt.show()
print(int(np.argmax(score)))

score_final=cross_val_score(KNN(4),X_dr,y,cv=5).mean()
print(f"PCA降维后KNN得分:{score_final}")

'''
效果可以：PCA降维后KNN得分:0.9688809523809525
'''