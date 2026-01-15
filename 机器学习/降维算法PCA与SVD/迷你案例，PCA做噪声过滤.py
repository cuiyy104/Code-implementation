from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

digits=load_digits()
print(digits.data.shape)
def plot_digits(data):
    fig,axes=plt.subplots(4,10,figsize=(10,4),
                          subplot_kw=dict(xticks=[], yticks=[]))
    for i,ax,in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),cmap='binary')
plot_digits(digits.data)
'''可以看出digits.data中有1797个样本，每个样本有64个特征，reshape之后对应一个8x8的图像'''
plt.show()

np.random.RandomState(42)

noisy=np.random.normal(digits.data,2)
'''
具体采样过程（逐元素采样）
对 digits.data 中的每一个元素，都执行以下操作：
以该元素的值为均值（比如 digits.data[0,0] = 0，则均值 = 0；digits.data[0,1] = 1，则均值 = 1）；
这里以 2 为标准差，从正态分布中随机抽取一个值；
把这个随机值放到 noisy 数组的相同位置。
'''
print(noisy.shape)
plot_digits(noisy)
plt.show()

pca=PCA(0.5).fit(noisy)
X_dr=pca.transform(digits.data)
without_noisy=pca.inverse_transform(X_dr)
plot_digits(without_noisy)
plt.show()
'''
可以看出，先保留50%的信息量，丢弃方差小的部分噪声，再还原回去之后，图像变得清晰了很多
'''