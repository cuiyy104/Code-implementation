from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

plus=KMeans(n_clusters=10).fit(X)
print(plus.n_iter_)

plus=KMeans(n_clusters=10,init='random',random_state=420).fit(X)
'''默认参数是init='k-means++'，即使用k-means++算法初始化聚类中心，效果更好一些'''
print(plus.n_iter_)

''' 参数max_iter：最大迭代次数，默认300,一般不需要调整'''
''' 参数tol：容忍度，默认1e-4,表示当簇中心的变化小于该值时停止迭代，一般不需要调整'''
random=KMeans(n_clusters=10,init='random',max_iter=10,random_state=420).fit(X)
y_pred_max10=random.labels_
print(f'得分： {silhouette_score(X,y_pred_max10)}')

random=KMeans(n_clusters=10,init='random',max_iter=20,random_state=420).fit(X)
y_pred_max20=random.labels_
print(f'得分： {silhouette_score(X,y_pred_max20)}')
'''基本没啥变化'''