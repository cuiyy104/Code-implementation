from matplotlib.pyplot import subplots
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X,y=make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)

fig,ax1=plt.subplots(1)
ax1.scatter(X[:,0],X[:,1],marker='o',s=8)
plt.show()

colors=['r','g','b','c']
fig,ax1=subplots(1)
for i in range(4):
    ax1.scatter(X[y==i,0],X[y==i,1],marker='o',s=8,color=colors[i])
plt.show()

n_clusters=4
cluster_=KMeans(n_clusters=n_clusters,random_state=0).fit(X)
print(cluster_.inertia_)
'''
kmeans属性：
cluster_.cluster_centers_  # 聚类中心坐标
cluster_.labels_           # 每个样本所属的簇标签
cluster_.inertia_         # 簇内误差平方和，越小越好
'''