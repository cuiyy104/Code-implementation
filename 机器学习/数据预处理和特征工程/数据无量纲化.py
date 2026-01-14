from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

'''数据归一化'''
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
pd.DataFrame(data)
scaler = MinMaxScaler()
scaler = scaler.fit(data)
result = scaler.transform(data)
print(result)
result = scaler.fit_transform(data) #等价于上面两步
print(scaler.inverse_transform(result)) #逆归一化操作

#当X中的特征数量非常多的时候，fit会报错并表示，数据量太大了我计算不了
#此时使用partial_fit作为训练接口
#scaler = scaler.partial_fit(data)

'''数据标准化'''
scaler = StandardScaler()
scaler = scaler.fit(data)
print(f"均值： {scaler.mean_},  方差： {scaler.var_}")
x_std=scaler.transform(data)
print(f"均值： {x_std.mean(axis=0)},  方差： {x_std.var(axis=0)}")
print(x_std)
'''一般常用数据标准化来处理数据'''