from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

'''属性components_'''

faces=fetch_lfw_people(min_faces_per_person=60)

'''
这里返回的faces是一个类似字典的对象，包含了数据和标签等信息
核心属性：
faces.images：原始人脸图像，形状为 (样本数, 62, 47)（每张图是 62 行 ×47 列的灰度像素）；
faces.data：将images扁平化后的一维数组，形状为 (样本数, 62×47=2914)（每张图的 2914 个像素排成一行，方便机器学习模型处理）；
faces.target：人物标签（数字），faces.target_names：标签对应的人物姓名
'''

print(faces.data.shape)
X=faces.data
fig,axes=plt.subplots(4,5,
                      figsize=(8,4),
                      subplot_kw={'xticks':[],'yticks':[]})

#使用迭代器
for i,ax in enumerate(axes.flat):
    ax.imshow(faces.images[i,:,:],cmap="gray")
'''上面的images本身就是一个二维图像，所以直接用imshow显示即可'''
plt.show()
pca=PCA(150).fit(X)
V=pca.components_
fig,axes=plt.subplots(3,8,figsize=(8,4),
                     subplot_kw={'xticks':[],'yticks':[]})
for i,ax in enumerate(axes.flat):
    ax.imshow(V[i,:].reshape(62,47),cmap="gray")
'''这里的components_属性返回的是每个主成分的方向向量，每个向量的长度等于原始数据的维度2914，
所以需要reshape成62行47列的二维图像才能显示出来'''
plt.show()