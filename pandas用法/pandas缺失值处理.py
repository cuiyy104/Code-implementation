import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fpath=r'titanic.csv'
pre_data=pd.read_csv(fpath)
print(np.all(pd.notnull(pre_data)))

data=pre_data[['age','survived']]
print(data.sample(10))

#检查是否有缺失值
print(np.all(pd.notnull(data)))
print(data.shape)
'''
删除缺失值函数dropna(axis,how,thresh,subset,inplace)
参数含义：
axis：0表示按行删除，1表示按列删除，默认值为0
how：any表示只要有缺失值就删除，all表示全部为缺失值才删除，默认值为any
thresh：表示非缺失值的数量阈值，默认值为None
subset：表示要检查的行或列的子集，默认值为None
inplace：表示是否在原数据上进行修改，True表示在原数据上修改，False表示返回一个新的数据，默认值为False
'''


'''删除缺失值'''
data=data.dropna(axis=0,how='any',thresh=None,inplace=False)
print(data.shape)
print(pre_data.shape)

'''保留至少14个非缺失值的行，就是说只要缺失值不是大于2的就不删除'''
del_data=pre_data.dropna(axis=0,thresh=14)
print(del_data.shape)
'''指定列删除'''
del_data=pre_data.dropna(axis=0,subset=['age'])#有两个字典就在中括号里面再写一个
print(del_data.shape)
'''删除包含缺失值的列'''
del_data=pre_data.dropna(axis=1)
print(del_data.shape)


'''
填充缺失值函数fillna(value,method,axis,inplace,limit)
参数含义：
value：表示用来填充缺失值的值，可以是一个标量值，也可以是一个字典，默认值为None
method：表示填充方法，pad表示用前一个值填充，bfill表示用后一个值填充，默认值为None
axis：表示填充的方向，0表示按行填充，1表示按列填充，默认值为0
inplace：表示是否在原数据上进行修改，True表示在原数据上修改，False表示返回一个新的数据，默认值为False
limit：表示填充的最大数量，默认值为None
'''

'''用固定值填充缺失值'''
fill_data=pre_data.fillna(value=0)
'''使用前一行数据进行填充'''
fill_data=pre_data.fillna(method='ffill')
'''使用后一行数据进行填充'''
fill_data=pre_data.fillna(method='bfill')
'''把axis改成1就是代表按后一列/前一列进行填充'''
print(fill_data.shape)
'''使用平均值进行填充'''
fill_data=pre_data.fillna(pre_data.mean())
'''使用中位数进行填充'''
fill_data=pre_data.fillna(pre_data.median())