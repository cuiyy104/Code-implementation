from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

'''
决策树分类器内部参数填写规则
criterion='gini'  # 信息增益衡量标准，默认'gini'基尼系数，'entropy'信息增益
splitter='best'   # 划分节点选择标准，默认'best'最佳划分，'random'随机划分
max_depth=None    # 树的最大深度，默认None不限制
min_samples_split=2  # 内部节点再划分所需最小样本数，默认2
min_samples_leaf=1   # 叶子节点最少样本数，默认1
max_features=None   # 划分节点时考虑的最大特征数，默认None所有特征
random_state=None   # 随机数种子，默认None不设置
class_weight=None  # 类别权重，默认None所有类权重相等,处理类别不平衡问题
ccp_alpha=0.0     # 复杂度惩罚参数，默认0.0不惩罚，用于剪枝
'''

wine = load_wine()
print(wine.data.shape)
print(wine.target)
print(wine.feature_names)
print(wine.target_names)
pdata=pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)

print(pdata.head())

Xtrain, Xtest, Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.2)
print(Xtrain.shape)
print(Xtest.shape)

clf=tree.DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(Xtrain,Ytrain)
score=clf.score(Xtest,Ytest)
print(score)

'''feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
dot_data=tree.export_graphviz(clf,
                              feature_names=feature_name,
                              class_names=['琴酒','白兰地','雪利酒'],
                              filled=True,rounded=True,
                              fontname='Microsoft YaHei'
                               )
graph=graphviz.Source(dot_data)
graph.render('wine_decision_tree',format='png')
graph.view()'''
clf2=tree.DecisionTreeClassifier(criterion='entropy',random_state=30)
clf2=clf2.fit(Xtrain,Ytrain)
score2=clf2.score(Xtest,Ytest)
print(score2)

