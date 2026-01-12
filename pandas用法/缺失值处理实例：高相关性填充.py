import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''
目标是计算样本之间的相关系数矩阵，
然后对含有缺失值的样本使用高相关性的样本进行填充
'''
fpath=r'titanic.csv'
pre_data=pd.read_csv(fpath).sample(100)#只取100行数据进行计算样本之间的相关性
'''筛选有数字的特征'''
numeric_data=pre_data.select_dtypes(include=[np.number]).columns
'''计算相关系数矩阵'''
#corr_matrix=numeric_data.T.corr()
'''
plt.figure(figsize=(10,10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    cbar=True,
    square=True,
    linewidths=0.5
)
plt.title('Correlation Matrix Heatmap', fontsize=20)
plt.show()'''

def fill_missing_values_using_correlation(data, threshold=0.8):
    filled = data.copy()
    # 只处理数值列
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    df_num = data[numeric_cols]

    missidx = df_num[df_num.isnull().any(axis=1)].index

    for idx in missidx:
        # 把当前样本做成一行 DataFrame
        curr_row = df_num.loc[idx].to_frame().T
        misscols = df_num.loc[idx][df_num.loc[idx].isnull()].index

        others = df_num.drop(idx)
        # 拼接后按列转置再计算样本间相关性
        sample_matrix = pd.concat([curr_row, others])
        sample_corr = sample_matrix.T.corr()

        curr_label = curr_row.index[0]
        # 取出与当前样本的相关系数（去掉自身）
        corr_with_curr = sample_corr.loc[curr_label].drop(curr_label)
        high_corr = corr_with_curr[corr_with_curr.abs() > threshold]

        if high_corr.empty:
            # 无高相关样本 -> 用整列均值填充
            for col in misscols:
                filled.loc[idx, col] = df_num[col].mean()
            continue

        # 用高相关样本在对应列的均值填充，若均值为 NaN 则回退到整列均值
        high_vals = others.loc[high_corr.index]
        for col in misscols:
            val = high_vals[col].mean()
            if np.isnan(val):
                val = df_num[col].mean()
            filled.loc[idx, col] = val

    return filled

# 示例调用（保持与原代码一致）
fpath = r'titanic.csv'
pre_data = pd.read_csv(fpath).sample(100)
filled_data = fill_missing_values_using_correlation(pre_data)

# 对比缺失值
numeric_before = pre_data.select_dtypes(include=[np.number]).isnull().sum()
numeric_after = filled_data.select_dtypes(include=[np.number]).isnull().sum()
print("\n填充前：\n", numeric_before)
print("填充后：\n", numeric_after)