from sko.ACA import ACA_TSP
import numpy as np
from scipy import spatial
import pandas as pd
import matplotlib.pyplot as plt

'''
蚁群算法是一种最优寻路算法，模拟蚂蚁在寻找食物过程中释放信息素并根据信息素浓度选择路径的行为。
'''

num_points = 50

# 用随机数生成 num_points 个点
points_coordinate = np.random.rand(num_points, 2)
# 调用 scipy 自动计算点与点之间的欧拉距离，生成距离矩阵
distance_matrix = spatial.distance.cdist(
    points_coordinate, points_coordinate, metric='euclidean')

def cal_total_distance(routine):
    num_points, = routine.shape
    # 计算距离和
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])

aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=450, max_iter=1000,
              distance_matrix=distance_matrix)
'''
函数的输入输出：
 input
func                # 目标函数，用一个参数 routine 来跟踪蚂蚁走过的路径    
n_dim               # 城市个数，也是 routine 参数的最终长度
size_pop            # 蚁群的规模
max_iter            # 最大迭代次数
distance_matrix     # 城市之间的距离矩阵，尽管 func 中已经根据这个矩阵求解路径距离总和，
                    # 仍然需要提供一个距离矩阵参数给算法以便计算信息素的挥发
alpha               # 信息素的重要程度，默认为 1
beta                # 启发值的重要程度，默认为 2
rho                 # 信息素的挥发速度，默认为 0.1，即信息素每次以 1-rho 的倍率衰减

# output
best_x              # 一个列表，记录最优的路径上的点的依序索引
best_y              # 一个值，记录最优路径的长度
y_best_history      # 一个表格，记录每次迭代求出的最优路径的值
                    # 横坐标是迭代次数索引，纵坐标最优路径值
'''
best_x, best_y = aca.run()
print('best_x: \n', best_x, '\n', 'best_y: ', best_y)

fig, ax = plt.subplots(1, 2)
best_circuit = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = points_coordinate[best_circuit, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r', markersize=3)
pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
plt.show()