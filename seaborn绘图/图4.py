import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="ticks")

dots = sns.load_dataset("dots")
print(dots.head())
print(dots.info())


# Define the palette as a list to specify exact values
palette = sns.color_palette("rocket_r")

# Plot the lines on two facets
sns.relplot(
    data=dots,
    x="time", y="firing_rate",
    hue="coherence", size="choice", col="align",
    kind="line", size_order=["T1", "T2"], palette=palette,
    height=5, aspect=.75, facet_kws=dict(sharex=False),
)
plt.show()

# python
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 加载并查看数据
df = sns.load_dataset("dots")
print(df.head())
print(df.info())

# 2D 趋势图（聚合为均值并显示置信区间）
sns.set_theme(style="ticks")
sns.relplot(
    data=df,
    x="time", y="firing_rate",
    hue="coherence", size="choice", col="align",
    kind="line", estimator=np.mean, ci=95,
    size_order=["T1", "T2"], palette="rocket_r",
    height=5, aspect=.75, facet_kws=dict(sharex=False),
)
plt.show()

# 3D 示例：Seaborn 不支持 3D，下面用 matplotlib 做简单的 3D 散点图
# 把类别编码为数值以便放到轴或颜色里
df3 = df.copy()
df3["coherence_code"] = df3["coherence"].astype("category").cat.codes
df3["choice_code"] = df3["choice"].astype("category").cat.codes

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    df3["time"], df3["coherence_code"], df3["firing_rate"],
    c=df3["coherence_code"], cmap="rocket_r",
    s=20 + df3["choice_code"] * 30, alpha=0.8
)
ax.set_xlabel("time")
ax.set_ylabel("coherence (coded)")
ax.set_zlabel("firing_rate")
plt.show()
