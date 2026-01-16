import networkx as nx

#无向图
G=nx.Graph()

#有向图
DG=nx.DiGraph()

#多重图
MG=nx.MultiGraph()

#有向多重图
MDG=nx.MultiDiGraph()

#添加节点
G.add_node(1) #添加节点1
G.add_nodes_from([2,3,4]) #添加多个节点2,3,4
G.add_node(5,lable='A',color='red') #添加节点5，并设置属性标签和颜色
G.add_nodes_from([(6,{'lable':'B','color':'blue'}),(7,{'lable':'C','color':'green'})]) #添加多个节点，并设置属性

#查看途中节点
nodes=G.nodes()
print(f"图的节点: {nodes}")
#查看节点属性
node_attrs=G.nodes[5]
print(f'节点5的属性: {node_attrs}')

#添加边
G.add_edge(1,2) #添加边(1,2)
G.add_edges_from([(2,3),(3,4)]) #添加多条边(2,3)和(3,4)
G.add_edge(4,5,weight=4.5) #添加边(4,5)，并设置权重属性
G.add_edges_from([(5,6,{'weight':2.5}),(6,7,{'weight':3.5})]) #添加多条边，并设置权重属性
G.add_edge(1,3,color='blue',weight=2.5)

#查看图的边
edges=G.edges()
print(f"所有边： {edges}")
#查看边属性
edge_attrs=G[4][5]
print(f'边(4,5)的属性: {edge_attrs}')

#访问图的元素
degree=G.degree(1)
print(f"节点1的度: {degree}")
neighbors=list(G.neighbors(2))
print(f"节点2的邻居: {neighbors}")
is_connected=nx.is_connected(G)
print(f"图是否连通: {is_connected}")
shortest_path=nx.shortest_path(G,source=1,target=5)
print(f"节点1到节点5的最短路径: {shortest_path}")
exists=G.has_node(5)
print(f"图中是否存在节点5: {exists}")
edge_exists=G.has_edge(2,3)
print(f"图中是否存在边(2,3): {edge_exists}")

#移除图中某些节点和边
G.remove_node(7) #移除节点7
G.remove_nodes_from([6,5]) #移除节点6和5
G.remove_edge(1,2) #移除边(1,2)
G.remove_edges_from([(2,3),(3,4)]) #移除边(2,3)和(3,4)

print(f"移除节点和边后图的节点: {G.nodes()}")
print(f"移除节点和边后图的边: {G.edges()}")