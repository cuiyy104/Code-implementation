import networkx as nx

G=nx.Graph()
G.add_nodes_from([1,2,3,4,5,6])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4), (5, 6)])

#给节点和边设置属性
G.nodes[1]['color']='red'
G.nodes[2]['color']='blue'
G.nodes[3]['color']='green'

G[1][2]['weight']=4.5
G[1][2]['label']='A to B'

#获取节点和边的属性
color=G.nodes[4].get('color','没有') #如果节点4没有颜色属性，则返回'没有'
print(f'节点1的颜色: {color}')

weight=G[1][2].get('weight',1.0) #如果边(1,2)没有权重属性，则返回1.0
print(f'边(1,2)的权重: {weight}')

#设置图的名称
G.graph['name']='My Graph'
print(f'图的名称: {G.graph["name"]}')

#获取所有节点和边的属性字典
node_attrs=nx.get_node_attributes(G,'color')
print(f'所有节点的颜色属性: {node_attrs}')
edge_attrs=nx.get_edge_attributes(G,'weight')
print(f'所有边的权重属性: {edge_attrs}')
