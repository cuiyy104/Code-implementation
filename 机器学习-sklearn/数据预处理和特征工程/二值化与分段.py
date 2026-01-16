from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pandas as pd

fpath=r'titanic.csv'
data=pd.read_csv(fpath)
#对年龄进行二值化处理
X=data.iloc[:,3].values.reshape(-1,1)
transform=Binarizer(threshold=30).fit_transform(X)
