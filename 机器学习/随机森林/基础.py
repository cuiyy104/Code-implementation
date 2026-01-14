from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from matplotlib import pyplot as plt
'''
建模随机森林流程

rfc = RandomForestClassifier()    #实例化                       

rfc = rfc.fit(X_train,y_train)    #训练                      

result = rfc.score(X_test,y_test)  #预测准确率                     

'''

wine=load_wine()
Xtrain,Xtest,Ytrain,Ytest=train_test_split(
    wine.data,
    wine.target,
    test_size=0.3,)
clf=DecisionTreeClassifier(random_state=0)
rfc=RandomForestClassifier(random_state=0)
clf.fit(Xtrain,Ytrain)
rfc.fit(Xtrain,Ytrain)
score_clf=clf.score(Xtest,Ytest)
score_rfc=rfc.score(Xtest,Ytest)
print(f'Decision Tree accuracy: {score_clf:.2f}')
print(f'Random Forest accuracy: {score_rfc:.2f}')
#随机森林的准确率通常会高于单一决策树的准确率，因为它通过集成多个决策树来减少过拟合和提高泛化能力。

'''使用交叉验证评估一下随机森林和决策树的性能'''
rfc_1=[]
clf_1=[]

for i in range(10):
    rfc=RandomForestClassifier(random_state=25)
    rfc_s=cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
    rfc_1.append(rfc_s)

    clf=DecisionTreeClassifier(random_state=25)
    clf_s=cross_val_score(clf,wine.data,wine.target,cv=10).mean()
    clf_1.append(clf_s)

plt.plot(range(1,11),rfc_1,label='Random Forest',marker='o')
plt.plot(range(1,11),clf_1,label='Decision Tree',marker='o')
plt.xlabel('Iteration')
plt.ylabel('Cross-Validated Accuracy')
plt.legend()
plt.show()

'''
可以看出随机森林在每次迭代中的交叉验证准确率都高于单一决策树，
说明随机森林在处理数据时具有更好的稳定性和泛化能力。
'''

'''试一下调参n_estimators'''
'''
n_estimators:
树木的数量，默认值为100。增加树的数量通常会提高模型的性能，
但也会增加计算成本。需要权衡性能和计算资源。
'''
superpa=[]
for i in range(1,11):
    j=i*10
    rfc=RandomForestClassifier(n_estimators=j,n_jobs=-1)
    rfc_s=cross_val_score(rfc,wine.data,wine.target,cv=10).mean()
    superpa.append(rfc_s)
print(max(superpa),superpa.index(max(superpa))*10)
plt.figure(figsize=[20,5])
x_labels=[i*10 for i in range(1,11)]
plt.plot(x_labels,superpa,label='Random Forest',marker='o')
plt.show()


'''
重要属性和接口
feature_importances_:
每个特征的重要性评分。可以用来评估哪些特征对模型的预测最有贡献。
oob_score_:
如果启用了袋外估计（oob_score=True），则该属性包含袋外样本的预测准确率。
estimators_:
森林中所有单个决策树的列表。可以用来访问和分析每棵树的细节。

随机森林的接口与决策树完全一致，因此依然有四个常用接口：apply, fit, predict和score
除此之外，还需要注
意随机森林的predict_proba接口，这个接口返回每个测试样本对应的被分到每一类标签的概率，标签有几个分类
就返回几个概率。如果是二分类问题，则predict_proba返回的数值大于0.5的，被分为1，小于0.5的，被分为0。
传统的随机森林是利用袋装法中的规则，平均或少数服从多数来决定集成的结果，而sklearn中的随机森林是平均
每个样本对应的predict_proba返回的概率，得到一个平均概率，从而决定测试样本的分类
'''

rfc = RandomForestClassifier(n_estimators=25)

rfc = rfc.fit(Xtrain, Ytrain)
print("------------------------------------------")
print(rfc.score(Xtest,Ytest))
print("------------------------------------------")
print(rfc.feature_importances_)
print("------------------------------------------")
print(rfc.apply(Xtest))
print("------------------------------------------")
print(rfc.predict(Xtest))
print("------------------------------------------")
print(rfc.predict_proba(Xtest))
