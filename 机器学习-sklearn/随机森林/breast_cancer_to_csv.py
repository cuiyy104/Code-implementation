# 将 sklearn 的乳腺癌数据集保存为 CSV 文件
import os
from sklearn.datasets import load_breast_cancer
import pandas as pd

# 加载数据集
data = load_breast_cancer()

# 构造 DataFrame：特征列 + 目标列（target）
X = pd.DataFrame(data.data, columns=data.feature_names)
X['target'] = data.target

# 输出路径（可按需修改）
out_dir = r"/机器学习-sklearn\随机森林"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'breast_cancer.csv')

# 保存为 CSV（不保留索引）
X.to_csv(out_path, index=False)

print(f"已保存：{out_path}")
print("数据集预览：")
print(X.head())
print(f"Shape: {X.shape}")

