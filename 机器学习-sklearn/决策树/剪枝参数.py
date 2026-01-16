from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
wine = load_wine()

Xtrain, Xtest, Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.2)
'''splitter 分裂策略参数'''
clf=tree.DecisionTreeClassifier(criterion='entropy',
                                random_state=30,
                                splitter='random')
clf=clf.fit(Xtrain,Ytrain)
score=clf.score(Xtest,Ytest)
print(score)
'''
splitter='random'表示在划分节点时随机选择特征和阈值，而不是选择最佳划分。这种方法增加了模型的随机性，
有助于减少过拟合，提升模型的泛化能力。
splitter='best'则表示每次划分节点时选择能够最大化信息增益（或最小化基尼系数）的特征和阈值，
通常会导致更深的树和更复杂的模型，可能更容易过拟合
'''

'''剪枝策略参数'''
#max_depth=None    # 树的最大深度，默认None不限制
#min_samples_split和min_samples_leaf
'''min_samples_split=2表示一个内部节点至少需要有2个样本才能继续划分。'''
clf2=tree.DecisionTreeClassifier(criterion='entropy',
                                 random_state=30,
                                 splitter='random',
                                 max_depth=3,
                                 min_samples_split=10,
                                 min_samples_leaf=10)
'''max_depth搭配min_samples_split和min_samples_leaf一起使用，
可以更有效地控制树的复杂度，防止过拟合，使模型更加平滑。'''
clf2=clf2.fit(Xtrain,Ytrain)
score2=clf2.score(Xtest,Ytest)
print(score2)

'''使用超参数曲线图来选择最佳剪枝参数'''
test=[]
for i in range(10):
    clf = tree.DecisionTreeClassifier(
        max_depth=i+1,
        criterion='entropy',
        random_state=30,
        splitter='random'
    )
    clf = clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest)
    test.append(score)
plt.plot(range(1,11),test,color='blue',label='maxdepth')
plt.legend()
plt.show()
'''通过观察图像，可以选择一个合适的max_depth值，'''

'''重要属性和接口'''
clf=tree.DecisionTreeClassifier(
    max_depth=4,
    criterion='entropy',
    random_state=30,
    splitter='random'
)
clf.fit(Xtrain,Ytrain)
score=clf.score(Xtest,Ytest)

'''apple返回每个测试样本所在叶子节点的索引'''
sample_idx=clf.apply(Xtest)

'''predict返回测试样本的预测类别'''
predicted_classes=clf.predict(Xtest)

'''predict_proba返回测试样本属于每个类别的概率'''
predicted_probabilities=clf.predict_proba(Xtest)

'''feature_importances_属性返回每个特征的重要性评分'''
feature_importances=clf.feature_importances_

print("Sample Indices:", sample_idx)
print("Predicted Classes:", predicted_classes)
print("Predicted Probabilities:", predicted_probabilities)
print("Feature Importances:", feature_importances)
