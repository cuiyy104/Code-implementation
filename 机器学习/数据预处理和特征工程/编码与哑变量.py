from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


'''
意思就是把文字性数据转换为数值型数据
'''

#preprocessing.LabelEncoder：标签专用，能够将分类转换为分类数值
fpath=r'titanic.csv'
data=pd.read_csv(fpath)

y=data.iloc[:,-1]
le=LabelEncoder()
le=le.fit(y)
labels=le.transform(y)
data.iloc[:,-1]=labels
print(data.head())
'''这样就把最后一列的alone情况转换为0和1了'''

#preprocessing.OrdinalEncoder：能够将多维的分类数据转换为多维的分类数值
data_=data.copy()
data_.iloc[:,1:-1]=OrdinalEncoder().fit_transform(data_.iloc[:,1:-1])
'''使用categories=[["S","C","Q"]]可以指定某一列的类别顺序'''
print(data_.head())
#这样就把多维的分类数据都转换为数值型数据了
# 注意LabelEncoder和OrdinalEncoder的区别
# LabelEncoder是针对一维数据进行转换的，而OrdinalEncoder是针对多维数据进行转换的

'''preprocessing.OneHotEncoder：独热编码，将分类数据转换为独热编码形式'''
X=[[1,'s'],[2,'m'],[3,'l']]
enc=OneHotEncoder(categories='auto').fit(X)
result=enc.transform(X).toarray()
print(result)