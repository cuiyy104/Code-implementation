from networkx.algorithms import community
import networkx as nx
import matplotlib.pyplot as plt

'''
社区发现的本质是基于图拓扑的节点聚类，核心目标是找到 “内紧外松” 的节点团体。
'''
# 1. 创建一个示例图（空手道俱乐部图）
G = nx.karate_club_graph()

# 2. 使用 Louvain 方法检测社区结构
#    louvain_communities 返回一个列表，列表中的每个元素是一个包含节点 id 的集合（set）
communities = community.louvain_communities(G, seed=42)

# 打印检测到的社区（每个社区是一组节点 id，例如 {0, 1, 2}）
print("检测到的社区:", communities)

# 3. 可视化社区结构：先计算布局（每个节点的 xy 坐标），再分别绘制每个社区的节点
#    spring_layout 使用弹簧模型进行力导向布局，使节点间具有较直观的分布。设置 seed 可复现布局。
pos = nx.spring_layout(G, seed=42)

# 颜色列表：为不同社区选不同颜色（如果社区数量超过颜色数，会循环使用）
colors = ['r', 'g', 'b', 'y', 'c', 'm']

# 4. 分别绘制每个社区的节点（不同颜色）并添加图例标签
#    enumerate 遍历 communities，i 为索引（从 0 开始），comm 为节点集合
for i, comm in enumerate(communities):
    # nodelist 参数接受可迭代节点集合；node_color 可以是单一颜色字符或与 nodelist 等长的颜色序列
    # 这里将集合转换为 list 以保证兼容性（虽然 networkx 也能接受 set）
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(comm),
        node_color=colors[i % len(colors)],  # 循环使用颜色列表
        label=f'community {i+1}'  # 给每个社区节点组设置标签，便于图例显示
    )

# 5. 绘制图中的所有边（半透明），这是在节点之下绘制的连接线
nx.draw_networkx_edges(G, pos, alpha=0.5)

# 6. 在节点上绘制节点标签（节点 id），便于查看每个节点属于哪个社区
nx.draw_networkx_labels(G, pos)

# 7. 设置图标题并显示图例
#    注意：由于我们为每个社区都添加了标签，plt.legend() 会把每个标签都显示出来；
#    如果多个社区颜色相同或想要合并图例项，可以在绘制时进行去重或手动构建 legend。
plt.title("Louvain community detection")
plt.legend()
plt.show()


#最小生成树
mst=nx.minimum_spanning_edges(G)
mst_edges=list(mst)
print("最小生成树的边:",mst_edges)
mst_graph=nx.Graph()
mst_graph.add_edges_from(mst_edges)
plt.figure()
nx.draw(mst_graph,pos,with_labels=True)
plt.title("Minimum Spanning Tree")
plt.show()

#最大流最小割
# 定义一个有向图
flow_G = nx.DiGraph()

# 添加带容量的边
flow_G.add_edge('A', 'B', capacity=15)
flow_G.add_edge('A', 'C', capacity=10)
flow_G.add_edge('B', 'D', capacity=10)
flow_G.add_edge('C', 'D', capacity=15)

#计算最大流，从A到D
flow_value,flow_dict=nx.maximum_flow(flow_G,'A','D')
print("最大流值:",flow_value)
print("流分布:",flow_dict)

