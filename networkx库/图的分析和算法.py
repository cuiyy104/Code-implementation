import networkx as nx

G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4, 5, 6])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3,{'weight': 4}), (2, 4), (5, 6)])

path = nx.shortest_path(G,source=1,target=4,weight='weight')
print("最短路径:", path)
path_len = nx.shortest_path_length(G,source=1,target=4,weight='weight')
print("最短长度:", path_len)

G.add_nodes_from([7,8,9])
# 计算图的连通分量
connected_components = list(nx.connected_components(G))
print("连通分量:", connected_components)
#查找某个点的连通分量
component=nx.node_connected_component(G,1)
print(f"节点1的连通分量: {component}")

G.remove_nodes_from([7,8,9])
G.add_edge(4,5)
# 计算图的直径
if nx.is_connected(G):
    diameter = nx.diameter(G)
    print("图的直径:", diameter)
else:
    print("图不是连通的，无法计算直径。")

# 计算图的聚类系数
'''
图的聚类系数衡量了图中节点的邻居之间相互连接的程度。它反映了节点的局部密集程度，值越高表示节点的邻居之间连接越紧密。
聚类系数的计算方法如下：
1. 对于图中的每个节点，计算其邻居节点之间实际存在的边数。
2. 计算该节点的邻居节点之间可能存在的最大边数。
3. 该节点的聚类系数等于实际边数除以可能边数。
4. 图的整体聚类系数可以通过计算所有节点的聚类系数的平均值来获得。
在NetworkX中，可以使用nx.clustering()函数计算单个节点的聚类系数，使用nx.average_clustering()函数计算整个图的平均聚类系数。
'''
clustering_coeffs = nx.clustering(G,1)
print(f"节点1的聚类系数: {clustering_coeffs}")
avg_clustering_coeff = nx.average_clustering(G)
print(f"图的平均聚类系数: {avg_clustering_coeff}")
all_clustering_coeffs = nx.clustering(G)
print("所有节点的聚类系数:", all_clustering_coeffs)

# 计算图的中心性指标
'''
中心性指标用于衡量图中节点的重要性或影响力。常见的中心性指标包括度中心性、接近中心性和介数中心性。
1. 度中心性（Degree Centrality）：衡量节点的连接数量，连接越多，度中心性越高。
2. 接近中心性（Closeness Centrality）：衡量节点与其他节点的平均距离，距离越短，接近中心性越高。
3. 介数中心性（Betweenness Centrality）：衡量节点在其他节点对之间的最短路径中出现的频率，频率越高，介数中心性越高。
在NetworkX中，可以使用nx.degree_centrality()，
                   nx.closeness_centrality()，
                   nx.betweenness_centrality()函数分别计算这些中心性指标。
'''

degree_centrality = nx.degree_centrality(G)
print("度中心性:", degree_centrality)
closeness_centrality = nx.closeness_centrality(G)
print("接近中心性:", closeness_centrality)
betweenness_centrality = nx.betweenness_centrality(G)
print("介数中心性:", betweenness_centrality)