from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''
在L1正则化在逐渐加强的过程中，携带信息量小的、对模型贡献不大的特征的参数，会比携带大量信息的、对模型
有巨大贡献的特征的参数更快地变成0，所以L1正则化本质是一个特征选择的过程，掌管了参数的“稀疏性”。L1正
则化越强，参数向量中就越多的参数为0，参数就越稀疏，选出来的特征就越少，以此来防止过拟合。因此，如果
特征量很大，数据维度很高，我们会倾向于使用L1正则化。由于L1正则化的这个性质，逻辑回归的特征选择可以由
Embedded嵌入法来完成。

相对的，L2正则化在加强的过程中，会尽量让每个特征对模型都有一些小的贡献，但携带信息少，对模型贡献不大
的特征的参数会非常接近于0。通常来说，如果我们的主要目的只是为了防止过拟合，选择L2正则化就足够了。但
是如果选择L2正则化后还是过拟合，模型在未知数据集上的效果表现很差，就可以考虑L1正则化。
'''

# 加载数据并随机抽样
data = load_breast_cancer()
X, y = data.data, data.target
rng = np.random.RandomState(42)                # 固定随机种子以便重现
idx = rng.choice(np.arange(X.shape[0]), size=50, replace=False)
X_s, y_s = X[idx], y[idx]

# PCA 降到 3 维（在抽样数据上拟合）
pca = PCA(n_components=3).fit(X_s)
X_pca = pca.transform(X_s)

# 可视化：3D 散点图，按类别分别绘制以显示图例
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for cls_idx, cls_name in enumerate(data.target_names):
    mask = (y_s == cls_idx)
    if mask.any():
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            X_pca[mask, 2],
            label=cls_name,
            s=50,
            edgecolor='k',
            alpha=0.9
        )

# 标注轴并显示每个主成分的解释方差百分比
evr = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f'PC1 ({evr[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({evr[1]:.1f}%)')
ax.set_zlabel(f'PC3 ({evr[2]:.1f}%)')
ax.set_title('随机 50 个样本的 PCA(3) 可视化')
ax.legend()
plt.tight_layout()
plt.show()
# 评估不同 penalty 和 C 对逻辑回归性能的影响
lrl1=LR(penalty='l1', solver='liblinear', C=0.5,max_iter=1000)
lrl2=LR(penalty='l2', solver='liblinear', C=0.5,max_iter=1000)

lrl1=lrl1.fit(X, y)
print(lrl1.score(X, y)) #得分
print(lrl1.coef_) #系数
print(lrl1.intercept_) #截距

lrl2=lrl2.fit(X, y)
print(lrl2.score(X, y))
print(lrl2.coef_)
print(lrl2.intercept_)

#对比一下正则化效果
l1=[]
l2=[]
l1test=[]
l2test=[]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=420)
for i in np.linspace(0.05,1,19):
    lrl1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
    lrl2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)

    lrl1 = lrl1.fit(X_train, Y_train)
    l1.append(accuracy_score(lrl1.predict(X_train), Y_train))
    l1test.append(accuracy_score(lrl1.predict(X_test), Y_test))

    lrl2 = lrl2.fit(X_train, Y_train)
    l2.append(accuracy_score(lrl2.predict(X_train), Y_train))
    l2test.append(accuracy_score(lrl2.predict(X_test), Y_test))

graph=[l1,l2,l1test,l2test]
color=['green','blue','red','black']
label=['train_l1','train_l2','test_l1','test_l2']
plt.figure(figsize=(6,6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05,1,19),graph[i],color=color[i],label=label[i])
plt.legend(loc=4)
plt.show()