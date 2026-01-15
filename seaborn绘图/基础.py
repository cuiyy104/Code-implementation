import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 模拟数据
years = np.arange(2010, 2021)
agri = np.array([200000,210000,220000,225000,230000,240000,250000,260000,270000,280000,290000])
hunt = np.array([15000]*11)
tour = np.array([30000,31000,32000,33000,34000,36000,38000,40000,42000,43000,44000])
eco = np.array([900000,890000,910000,920000,940000,960000,980000,1000000,1010000,1020000,990000])

df = pd.DataFrame({
    'year': years,
    'Agricultural': agri,
    'Hunting': hunt,
    'Tourism': tour,
    'Ecological': eco
})

# 绘图
fig, ax = plt.subplots(figsize=(10,6))

x = np.arange(len(df))
width = 0.7

# 先画底部类，依次堆叠
p1 = ax.bar(x, df['Agricultural'], width, label='Agricultural added value', color='#fff2a8', edgecolor='k')
p2 = ax.bar(x, df['Hunting'], width, bottom=df['Agricultural'], label='Hunting industry output value', color='#f28b82', edgecolor='k')
p3 = ax.bar(x, df['Tourism'], width, bottom=df['Agricultural']+df['Hunting'],
            label='Tourism revenues', color='#c7f0e8', edgecolor='k', hatch='..')
p4 = ax.bar(x, df['Ecological'], width, bottom=df['Agricultural']+df['Hunting']+df['Tourism'],
            label='Ecological output value', color='#d4d6ff', edgecolor='k', hatch='////')

# x 轴标签为年份
ax.set_xticks(x)
ax.set_xticklabels(df['year'], rotation=0)

# y 轴数值格式（科学计数或千分位）
ax.set_ylabel('Total economic benefits')
ax.set_title('Stacked economic benefits by year')

# 自定义图例（包括图案）
legend_handles = [
    Patch(facecolor='#fff2a8', edgecolor='k', label='Agricultural added value'),
    Patch(facecolor='#f28b82', edgecolor='k', label='Hunting industry output value'),
    Patch(facecolor='#c7f0e8', edgecolor='k', hatch='..', label='Tourism revenues'),
    Patch(facecolor='#d4d6ff', edgecolor='k', hatch='////', label='Ecological output value')
]
ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))



plt.tight_layout()
plt.savefig('stacked_example.png', dpi=300, bbox_inches='tight')  # 保存高分辨率
plt.show()