import numpy as np
import pandas as pd

sdata = pd.Series(np.arange(1,4),index=list('abc'))
'''print(sdata)
print(sdata.iloc[0])
print(sdata['b'])
print(sdata.loc['c'])
print(sdata.index)
print(sdata.values)
'''

data=np.arange(16).reshape(4,4)
data=pd.DataFrame(data,index=list('abcd'),columns=list('WXYZ'))
print(data)

pdata={'c1':[1,2,3],'c2':['a','b','c'],'c3':[4,5,6]}
pdata=pd.DataFrame(pdata)
print(pdata)

print(pdata['c2'][1])
print(pdata[['c1','c2']])
#遍历
for col in pdata:
    for val in pdata[col]:
        print(val)

#按列遍历
for val,item in pdata.items():
    print(val)
    print(item[0],item[1],item[2])
#按行遍历
for index,row in pdata.iterrows():
    print(index)
    print(row['c1'],row['c2'],row['c3'])

#插入列
pdata['c4']=[-1,-2,-3]
print(pdata)

pdata.loc[3]=[4,'d',7,-4]
print(pdata)

'''保存'''
pdata.to_csv(r'test.csv')