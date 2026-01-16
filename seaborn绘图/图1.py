import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
sns.set_theme(style="ticks")



# Load the example dataset for Anscombe's quartet
df = sns.load_dataset("anscombe")

print("数据head")
print(df.head())

print("数据内容")
print(df.info())

# Show the results of a linear regression within each dataset
sns.lmplot(
    data=df, x="x", y="y", col="dataset", hue="dataset",
    col_wrap=2, palette="muted", ci=None,
    height=4, scatter_kws={"s": 50, "alpha": 1}
)
plt.show()
'''
lmplot 是一个 figure\-level 的函数（内部用 `regplot` + `FacetGrid`），用于在分面上绘制带拟合直线的散点图。下面简短说明常用参数（含你代码里的那些）：

- `data`：要绘图的 `DataFrame`。  
- `x`, `y`：要绘制的列名（横纵坐标）。  
- `hue`：按该列分色，给不同组不同颜色并可绘制图例。  
- `col` / `row`：按该列做列/行分面（每个子图是一组）。  
- `col_wrap`：当 `col` 太多时，按指定列数换行（如 `2` 表示每行 2 列）。  
- `palette`：调色板，控制 `hue` 各类的颜色（如 `"muted"`）。  
- `ci`：置信区间宽度（默认 `95`）；设为 `None` 关闭置信区间；也可设数字或 `'sd'`。  
- `height`：每个子图的高度（英寸）。  
- `aspect`：子图宽高比（宽 = `height * aspect`）。  
- `scatter_kws` / `line_kws`：传给散点 / 拟合线的额外绘图参数（字典），例如 `{"s":50, "alpha":0.8}`。  
- `order`：拟合多项式的阶数（默认 `1`，即线性）。  
- `lowess`：若 `True` 用局部加权回归平滑（非参数）。  
- `markers`：指定散点的标记样式（可为单个或按 `hue` 的列表）。  
- `sharex` / `sharey`：各分面是否共享 x/y 轴范围（`True`/`False`）。  
- `truncate`：是否把拟合线裁剪到数据范围内（`True`/`False`）。

简短提示：`lmplot` 适合快速按组可视化带回归线的关系；要更细粒度控制图形元素，可以用 `regplot`（axes\-level）或直接操作 `FacetGrid`。
'''

'''如果有z轴，手动拓展成三维'''
df = sns.load_dataset("anscombe")
np.random.seed(0)
df['z'] = 0.5 * df['x'] + 0.3 * df['y'] + np.random.normal(scale=1.0, size=len(df))

datasets = sorted(df['dataset'].unique())
fig = plt.figure(figsize=(10, 8))

for i, name in enumerate(datasets):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    sub = df[df['dataset'] == name]
    x = sub['x'].values
    y = sub['y'].values
    z = sub['z'].values

    # 3D 散点
    ax.scatter(x, y, z, s=40, alpha=0.9)

    # 拟合平面 z = a*x + b*y + c
    A = np.c_[x, y, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)  # coef = [a, b, c]

    # 在数据范围上构造网格并计算平面高度
    xx, yy = np.meshgrid(
        np.linspace(x.min(), x.max(), 10),
        np.linspace(y.min(), y.max(), 10)
    )
    zz = coef[0] * xx + coef[1] * yy + coef[2]

    # 绘制平面
    ax.plot_surface(xx, yy, zz, color='C0', alpha=0.3, linewidth=0, antialiased=True)

    ax.set_title(f"dataset = {name}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

plt.tight_layout()
plt.show()