import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
sns.set_theme(style="whitegrid")

# Load the example diamonds dataset
diamonds = sns.load_dataset("diamonds")

print("数据head")
print(diamonds.head())
print("数据内容")
print(diamonds.info())
# Draw a scatter plot while assigning point colors and sizes to different
# variables in the dataset
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
sns.scatterplot(x="carat", y="price",
                hue="clarity", size="depth",
                palette="ch:r=-.2,d=.3_r",
                hue_order=clarity_ranking,
                sizes=(1, 8), linewidth=0,
                data=diamonds, ax=ax)
plt.show()
'''
data
类型：DataFrame 或 dict 或数组
说明：数据源，若使用列名（x, y, hue...）则必须传 DataFrame。
示例：data=diamonds
x, y
类型：字符串（列名）或一维数组
说明：横纵坐标变量，若使用列名需存在于 data 中。
示例：x="carat", y="price"
hue
类型：字符串（分类或连续列名）或数组
说明：按该变量分色。若为分类，会为每类分配颜色；若为连续，会使用渐变色。
注意：若想固定颜色顺序，配合 hue_order 使用。
示例：hue="clarity"
hue_order
类型：序列
说明：指定 hue 类别的显示/配色顺序（必须与 hue 的唯一值匹配）。
示例：hue_order=["I1","SI2",...]
palette
类型：字符串（预设调色板名）、列表或 dict 或 seaborn 色码表达式
说明：控制颜色。可以使用预设（"viridis"、"muted"）、列表（每类一个颜色）或像你示例中的 ch:...（cubehelix 语法）。
示例：palette="ch:r=-.2,d=.3_r" 或 palette=["#1f77b4", "#ff7f0e"]
size
类型：字符串（列名）或数组
说明：按该变量调整点大小。可为离散或连续变量。
注意：和 sizes 一起使用以控制实际像素范围；离散 size 可配 size_order。
示例：size="depth"
sizes
类型：二元元组或序列
说明：当 size 为连续变量时，通常传 (min_size, max_size) 控制映射范围；当 size 为分类变量时，也可以传一个与类别数相同的序列，分别指定每类的大小。
示例：sizes=(10, 200) 或 sizes=[20, 50, 80]
size_order
类型：序列
说明：指定离散 size 的类别顺序（与 hue_order 类似）。
style
类型：字符串（列名）或列表
说明：按类别改变 marker 样式（"o", "X" 等），适合同时编码第三个分类变量。
示例：style="cut"
markers
类型：布尔、字典或序列
说明：是否使用不同标记或为每类指定标记样式。markers=True 自动；也可传字典映射类别到样式。
示例：markers=True 或 markers={"A":"o","B":"s"}
palette / hue_norm / size_norm
说明：hue_norm/size_norm 可用于标准化连续变量的色/尺寸映射（传 matplotlib.colors.Normalize）。
alpha
类型：浮点数 0-1
说明：点透明度，常用于减轻重叠影响。
示例：alpha=0.6
linewidth / edgecolor / facecolor
说明：控制点边缘宽度与颜色（linewidth=0 常用于去掉边缘）。
legend
类型："brief" / "full" / False
说明：是否显示图例及其详略级别。
示例：legend="brief"
ax
类型：matplotlib Axes
说明：将图画在指定坐标轴上，用于在子图或自定义布局中复用。
示例：ax=ax
palette 字符串规则（例如 ch:r=-.2,d=.3_r）
说明：这是 cubehelix 调色板的短语法，按需用即可；一般使用常见名更直观（"viridis"、"coolwarm"、"muted" 等）。
'''

sns.set_theme(style="whitegrid")

# 加载数据并可选抽样（加速绘图）
diamonds = sns.load_dataset("diamonds")
diamonds = diamonds.sample(n=3000, random_state=0)  # 若数据量小可删除这行

# 变量映射
x = diamonds["carat"].values
y = diamonds["price"].values
z = diamonds["depth"].values    # 第三维，亦可用 diamonds["table"]
size_col = diamonds["table"].values  # 用 table 控制点大小（示例）
clarity = diamonds["clarity"].values

# clarity 的颜色映射（与原来 hue_order 保持一致）
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
palette = sns.color_palette("ch:r=-.2,d=.3_r", n_colors=len(clarity_order))
color_map = {k: palette[i] for i, k in enumerate(clarity_order)}
colors = [color_map[c] for c in clarity]

# 将 size_col 映射到点面积（matplotlib 的 s 是像素面积）
s_min, s_max = 10, 200
sizes = np.interp(size_col, (size_col.min(), size_col.max()), (s_min, s_max))

# 绘制 3D 散点
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.7, edgecolors="w", linewidth=0.2)

ax.set_xlabel("carat")
ax.set_ylabel("price")
ax.set_zlabel("depth")  # 若用 table 则改为 "table"
ax.view_init(elev=20, azim=-60)

# clarity 图例（颜色）
clarity_handles = [
    mlines.Line2D([], [], color=color_map[c], marker="o", linestyle="",
                  markersize=8, label=c)
    for c in clarity_order if c in diamonds["clarity"].unique()
]
legend1 = ax.legend(handles=clarity_handles, title="clarity",
                    bbox_to_anchor=(1.02, 1), loc="upper left")

# size 图例（点大小）
size_vals = np.percentile(size_col, [10, 50, 90])
size_handles = [
    plt.scatter([], [], s=np.interp(v, (size_col.min(), size_col.max()), (s_min, s_max)),
                color="gray", alpha=0.7, edgecolors="w")
    for v in size_vals
]
labels = [f"{int(v)}" for v in size_vals]
legend2 = ax.legend(size_handles, labels, title="table3D",
                    bbox_to_anchor=(1.02, 0.5), loc="center left")
ax.add_artist(legend1)

plt.tight_layout()
plt.show()