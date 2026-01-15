import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris=load_iris()
y=iris.target
X=iris.data

print(X.shape)
print(pd.DataFrame(X))
print(iris.feature_names)
print(iris.target_names)
pca=PCA(n_components=2)
pca=pca.fit(X)
X_dr=pca.transform(X)
#print(x_dr)

'''可视化展示'''
plt.figure()
plt.scatter(X_dr[y==0, 0], X_dr[y==0, 1], c="red", label=iris.target_names[0])
plt.scatter(X_dr[y==1, 0], X_dr[y==1, 1], c="black", label=iris.target_names[1])
plt.scatter(X_dr[y==2, 0], X_dr[y==2, 1], c="orange", label=iris.target_names[2])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()

'''探索降维之后的数据'''
print(pca.explained_variance_ratio_) #各主成分的方差百分比,也可以视为各主成分的重要性
'''通常情况下，第一个特征向量对应的方差最大，第二个特征向量对应的方差次之，依此类推。'''
print(pca.explained_variance_)       #各主成分的方差值
print(pca.components_)               #各主成分的方向向量

'''选择n_components的超参数验证曲线'''
'''实际上因为PCA是线性降维算法，所以n_components的选择不会影响降维后的结果，只会影响降维后数据的维度'''
'''所以可以让PCA维度不变，直接画出前k个主成分的累计方差贡献率曲线，然后以此为依据选择n_components的值'''
pca_line=PCA().fit(X)
plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1,2,3,4])
plt.xlabel('n_components')
plt.ylabel('cumulative explained_variance')
plt.show()

'''最大似然估计选择'''
pca_mle=PCA(n_components='mle').fit(X)
X_mle=pca_mle.transform(X)
print(X_mle)
print(pca_mle.explained_variance_ratio_.sum)

'''按照信息量占比来选择超参数'''
pca_f=PCA(n_components=0.95,svd_solver='full').fit(X)
'''意思是希望PCA降维之后的数据能够保留95%的信息量'''
X_f=pca_f.transform(X)
print(X_f)
print(pca_f.explained_variance_ratio_.sum())