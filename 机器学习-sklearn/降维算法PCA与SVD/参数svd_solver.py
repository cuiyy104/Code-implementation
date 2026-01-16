import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
'''
参数svd_solver有四种选择模式

auto:基于X.shape和n_components的默认策略来选择分解器：如果输入数据的尺寸大于500x500且要提
取的特征数小于数据最小维度min(X.shape)的80％，就启用效率更高的”randomized“方法。否则，精确完整
的SVD将被计算，截断将会在矩阵被分解完成后有选择地发生

full:：从scipy.linalg.svd中调用标准的LAPACK分解器来生成精确完整的SVD，适合数据量比较适中，计算时
间充足的情况

arpack:可以加快运算速度，适合特征矩阵很大的时候，但一般用于
特征矩阵为稀疏矩阵的情况，此过程包含一定的随机性

randomized:适合特征矩阵巨大，计算量庞大的情况。

而参数random_state在参数svd_solver的值为"arpack" or "randomized"的时候生效，可以控制这两种SVD模式中
的随机模式。通常我们就选用”auto“，不必对这个参数纠结太多
'''