import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
# 加载 fmri 数据集
fmri = sns.load_dataset("fmri")

# 显示数据集的前 5 行
print("数据集前 5 行：")
print(fmri.head())

# （可选）显示数据集的基本信息，了解变量类型和样本量
print("\n数据集基本信息：")
fmri.info()

# （可选）显示分组变量的唯一值，确认分组情况
print("\n脑区（region）的唯一值：", fmri["region"].unique())
print("事件类型（event）的唯一值：", fmri["event"].unique())

# 导入需要的库

# 设置绘图主题（和原图一致）
sns.set_theme(style="darkgrid")

# 加载 fmri 数据集
fmri = sns.load_dataset("fmri")

# 绘制带误差带的分组折线图（参数完全对应原图）
sns.lineplot(
    x="timepoint",    # 横轴：时间点
    y="signal",       # 纵轴：信号强度
    hue="region",     # 颜色区分：脑区（parietal/frontal）
    style="event",    # 线条样式区分：事件类型（stim/cue）
    data=fmri
)

# 显示图形
plt.title("Timeseries plot with error bands")  # 添加和原图一致的标题
plt.show()

'''改三维'''

fmri = sns.load_dataset("fmri")
grp = fmri.groupby(['timepoint', 'region', 'event'])['signal'].agg(['mean', 'sem']).reset_index()
grp['ci'] = 1.96 * grp['sem']  # 近似 95% 置信区间

# 把 region 映射到 y 轴数值，并为不同 event 加小偏移以便区分
regions = sorted(grp['region'].unique())
region_pos = {r: i for i, r in enumerate(regions)}
grp['y_base'] = grp['region'].map(region_pos)
event_offsets = {'stim': -0.12, 'cue': 0.12}
grp['y'] = grp['y_base'] + grp['event'].map(event_offsets)

# 绘图：每个 (region, event) 一条 3D 折线，置信区间用竖直线段表示
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

colors = sns.color_palette("tab10", n_colors=len(regions))
color_map = {r: colors[i] for i, r in enumerate(regions)}

for (region, event), sub in grp.groupby(['region', 'event']):
    x = sub['timepoint'].values
    y = sub['y'].values
    z = sub['mean'].values
    ci = sub['ci'].values

    linestyle = '-' if event == 'stim' else '--'
    ax.plot(x, y, z, color=color_map[region], linestyle=linestyle, label=f"{region} / {event}")

    # 每个点画一条竖直误差线表示置信区间
    for xi, yi, zi, cii in zip(x, y, z, ci):
        ax.plot([xi, xi], [yi, yi], [zi - cii, zi + cii], color=color_map[region], alpha=0.4)

# 美化与图例
ax.set_xlabel('timepoint')
ax.set_ylabel('region (mapped)')
ax.set_zlabel('signal')
ax.set_yticks(list(region_pos.values()))
ax.set_yticklabels(list(region_pos.keys()))
ax.view_init(elev=25, azim=-60)

# 单独创建 region（颜色）和 event（线型）图例
region_handles = [Line2D([0], [0], color=color_map[r], lw=3) for r in regions]
event_handles = [Line2D([0], [0], color='k', lw=2, linestyle='-'),
                 Line2D([0], [0], color='k', lw=2, linestyle='--')]
leg1 = ax.legend(region_handles, regions, title='region', bbox_to_anchor=(1.02, 1), loc='upper left')
leg2 = ax.legend(event_handles, ['stim', 'cue'], title='event', bbox_to_anchor=(1.02, 0.6), loc='upper left')
ax.add_artist(leg1)

plt.tight_layout()
plt.show()