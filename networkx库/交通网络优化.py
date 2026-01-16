import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
G = nx.DiGraph()

# 添加节点（路口）
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
G.add_nodes_from(nodes)

# 添加带权重的边（道路）及其通行能力
edges = [
    ('A', 'B', {'weight': 2, 'capacity': 100}),
    ('A', 'C', {'weight': 5, 'capacity': 50}),
    ('B', 'C', {'weight': 2, 'capacity': 100}),
    ('B', 'D', {'weight': 4, 'capacity': 70}),
    ('C', 'E', {'weight': 1, 'capacity': 120}),
    ('D', 'F', {'weight': 3, 'capacity': 80}),
    ('E', 'F', {'weight': 1, 'capacity': 150}),
    ('F', 'G', {'weight': 2, 'capacity': 90}),
    ('E', 'H', {'weight': 3, 'capacity': 60}),
    ('H', 'G', {'weight': 2, 'capacity': 70}),
]

G.add_edges_from([(u, v, data) for u, v, data in edges])

# 可视化交通网络（美化）
# - 美化：节点边框、边宽与权重成比例、边略微弯曲以减少重叠、标签带白色背景框、添加图例、隐藏坐标轴
pos = nx.spring_layout(G, seed=42)  # 固定 seed 以复现布局
plt.figure(figsize=(10, 8))

# 节点绘制（带黑色边框，使节点更清晰）
node_colors = 'lightblue'
nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, edgecolors='black')
# 节点标签（节点名字，如 'A','B'）
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# 边绘制：宽度与 weight 成正比，使用略弯曲的 connectionstyle 减少重叠；箭头样式美化有向图
edge_weights = [data.get('weight', 1) for _, _, data in G.edges(data=True)]
# 适当缩放边宽以适配显示（可按需调整缩放因子）
edge_widths = [max(0.8, w * 0.8) for w in edge_weights]
# 逐条绘制边，以便为每条边传入单个浮点类型的 width（消除类型检查警告）
for (u, v, data), w in zip(G.edges(data=True), edge_widths):
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v)],
        arrowstyle='->',
        arrowsize=20,
        width=float(w),
        edge_color='gray',
        connectionstyle='arc3,rad=0.08'  # 弧度值可调整，正负表示左右弯曲
    )

# 边标签（显示 weight/capacity），带白色半透明背景提高可读性
edge_labels = {
    (u, v): f"{data.get('weight', '')}/{data.get('capacity', '')}"
    for u, v, data in G.edges(data=True)
}
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    label_pos=0.5,
    font_size=9,
    font_color='black',
    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8),
    rotate=False,
)

# 添加图例：节点与边的说明（自定义图例句柄）
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
node_patch = mpatches.Patch(color=node_colors, label='路口（节点）')
edge_line = Line2D([0], [0], color='gray', lw=2, label='道路（宽度 ∝ 权重）', marker='>', markersize=8)
plt.legend(handles=[node_patch, edge_line], loc='upper left')

plt.title('城市交通网络示意图')
plt.axis('off')  # 关闭坐标轴显示
plt.tight_layout()
plt.show()

#1.找最短路
shortest_path = nx.shortest_path(G, source='A', target='G', weight='weight')
shortest_path_length = nx.shortest_path_length(G, source='A', target='G', weight='weight')
print(f"从A到G的最短路径为：{shortest_path}，总时间为：{shortest_path_length}")

#2.找到最大流量路径
max_flow_value, max_flow_dict = nx.maximum_flow(G, 'A', 'G',    capacity='capacity')
print(f"从A到G的最大流量为：{max_flow_value}")
print(f"最大流路径分配为：{max_flow_dict}")

#3.避免拥堵的路径（可以通过增加某些边的权重来模拟拥堵
# 我们假设边 ('B', 'C') 和 ('E', 'F') 在高峰期很拥堵，因此增加它们的权重
G['B']['C']['weight'] += 10
G['E']['F']['weight'] += 10

# 重新计算最短路径
congestion_avoidance_path = nx.shortest_path(G, source='A', target='G', weight='weight')
congestion_avoidance_path_length = nx.shortest_path_length(G, source='A', target='G',
                                                           weight='weight')
print(f"避开拥堵后从A到G的路径为：{congestion_avoidance_path}",
      f"总时间为：{congestion_avoidance_path_length}", sep='，')
