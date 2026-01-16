import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier

x,y=make_classification(
    n_samples=100, # 样本数量
    n_features=2, # 特征数量
    n_redundant=0, # 冗余特征数量
    n_informative=2, # 信息特征数量
    random_state=1, # 随机种子
    n_clusters_per_class=1 # 每类簇的标签数
)
'''
x是100行2列的特征矩阵，y是对应的类别标签（0或1）。
'''
#plt.scatter(x[:, 0], x[:, 1])
#plt.show()

'''打乱一下特征的分布，x的特征加减0-2之间的随机噪声'''
rnp =np.random.RandomState(2)
x+=2*rnp.uniform(size=x.shape)
linearly_separable=(x,y)
#plt.scatter(x[:, 0], x[:, 1],c=y)
#plt.show()

#用make_moons创建月亮型数据，make_circles创建环形数据，并将三组数据打包起来放在列表datasets中
datasets=[make_moons(noise=0.3, random_state=0),
         make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

figure=plt.figure(figsize=(6,9))
i=1

for ds_idx,ds in enumerate(datasets):

    '''进行标准化处理，然后划分训练集和测试集'''
    x,y=ds
    x=StandardScaler().fit_transform(x)
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=.4,random_state=42)

    '''找到x的两个特征的最大最小值，用于后续绘制决策边界'''
    x1_min,x1_max=x[:,0].min()-.5,x[:,0].max()+.5
    x2_min,x2_max=x[:,1].min()-.5,x[:,1].max()+.5

    array1,array2=np.meshgrid(np.arange(x1_min,x1_max,.2),
                              np.arange(x2_min,x2_max,.2))
    #生成彩色画布
    cm=plt.cm.RdBu
    cm_bright =ListedColormap(['r','g','b'])

    ax=plt.subplot(len(datasets),2,i)
    if ds_idx == 0:
        ax.set_title("Input data")
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                   cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
               cmap=cm_bright, alpha=0.6, edgecolors='k')
    ax.set_xlim(array1.min(), array1.max())
    ax.set_ylim(array2.min(), array2.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    '''上面代码是绘制输入数据，下面代码是训练决策树并绘制决策边界'''

    # 迭代决策树，首先用subplot增加子图，subplot(行，列，索引)这样的结构，并使用索引i定义图的位置
    # 在这里，len(datasets)其实就是3，2是两列
    # 在函数最开始，我们定义了i=1，并且在上边建立数据集的图像的时候，已经让i+1,所以i在每次循环中的取值是2，4，6
    ax=plt.subplot(len(datasets),2,i)

    # 决策树的建模过程：实例化 → fit训练 → score接口得到预测的准确率
    clf=DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train,y_train)
    score=clf.score(X_test,y_test)
    '''预测整个网格点的类别'''
    # 绘制决策边界，为此，我们将为网格中的每个点指定一种颜色[x1_min，x1_max] x [x2_min，x2_max]
    # 分类树的接口，predict_proba，返回每一个输入的数据点所对应的标签类概率
    # 类概率是数据点所在的叶节点中相同类的样本数量/叶节点中的样本总数量
    #  由于决策树在训练的时候导入的训练集X_train里面包含两个特征，所以我们在计算类概率的时候，也必须导入
    #结构相同的数组，即是说，必须有两个特征
    # ravel()能够将一个多维数组转换成一维数组
    # np.c_是能够将两个数组组合起来的函数
    #  在这里，我们先将两个网格数据降维降维成一维数组，再将两个数组链接变成含有两个特征的数据，再带入决策
    #树模型，生成的Z包含数据的索引和每个样本点对应的类概率，再切片，且出类概率
    Z=clf.predict_proba(np.c_[array1.ravel(),array2.ravel()])[:,1]
    # 将返回的类概率作为数据，放到contourf里面绘制去绘制轮廓
    Z=Z.reshape(array1.shape)
    ax.contourf(array1,array2,Z,cmap=cm,alpha=.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
               cmap=cm_bright, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
               cmap=cm_bright, edgecolors='k', alpha=0.6)
    ax.set_xlim(array1.min(), array1.max())
    ax.set_ylim(array2.min(), array2.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_idx == 0:
        ax.set_title("Decision Tree (max_depth=5)\nAccuracy: %.2f" % score)
    ax.text(array1.max() - .3, array2.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()

'''
从图上来看，每一条线都是决策树在二维平面上画出的一条决策边界，每当决策树分枝一次，就有一条线出现。当
数据的维度更高的时候，这条决策边界就会由线变成面，甚至变成我们想象不出的多维图形。

同时，很容易看得出，分类树天生不擅长环形数据。每个模型都有自己的决策上限，所以一个怎样调整都无法提升
表现的可能性也是有的。当一个模型怎么调整都不行的时候，我们可以选择换其他的模型使用，不要在一棵树上吊
死。顺便一说，最擅长月亮型数据的是最近邻算法，RBF支持向量机和高斯过程；最擅长环形数据的是最近邻算法
和高斯过程；最擅长对半分的数据的是朴素贝叶斯，神经网络和随机森林。
'''