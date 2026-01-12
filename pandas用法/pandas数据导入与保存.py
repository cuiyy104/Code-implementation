import pandas as pd
import numpy as np

fpath=r'titanic.csv'
pdata=pd.read_csv(fpath,header=None)
#print(pdata.shape)
print(pdata.loc[0])
print(pdata.sample(10))
'''只读取某一列'''
data=pd.read_csv(fpath,usecols=['survived'])
#print(data.shape)
#print(data.sample(10))

