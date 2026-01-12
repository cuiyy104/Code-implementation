import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA

"""
遗传算法优劣势
优势：
1. 全局搜索能力强：遗传算法通过群体进化和遗传操作，能够有效探索解空间，避免陷入局部最优解。
2. 适应性强：遗传算法可以处理各种类型的优化问题，包括连续和离散优化问题。
3. 并行计算能力：遗传算法天然适合并行计算，可以利用多核处理器提高计算效率。
4. 不依赖梯度信息：遗传算法不需要目标函数的导数信息，适用于不可导或复杂的函数。
5. 多样性维护：通过交叉和变异操作，遗传算法能够保持种群的多样性，增强搜索能力。
劣势：
1. 收敛速度较慢：遗传算法的收敛速度可能较慢，尤其是在复杂的高维问题中。
2. 参数敏感性：遗传算法的性能对参数设置较为敏感，不同参数组合可能导致截然不同的结果。
3. 计算资源消耗大：遗传算法通常需要较多的计算资源，尤其是在大规模问题中。
4. 缺乏理论支持：相比其他优化算法，遗传算法缺乏系统的理论分析，导致其性能难以预测。
5. 适用范围有限：遗传算法在某些特定类型的问题上表现不佳，如高度非线性或多峰问题。
"""

# 背包问题参数
weight = np.array([2, 3, 4, 5, 6])  # 物品重量
value = np.array([3, 4, 5, 6, 7])   # 物品价值
max_weight = 10                     # 最大承重

# 1. 定义目标函数（最大化总价值，需返回负值让GA最小化）
def knapsack_func(x):
    # x是0-1数组，x[i]=1表示选第i个物品，x[i]=0表示不选
    total_weight = np.sum(x * weight)
    total_value = np.sum(x * value)
    # 约束：超重则惩罚（价值置0）
    if total_weight > max_weight:
        return 0  # GA最小化，返回0相当于惩罚（原本要最大化价值）
    return -total_value  # 取负，让GA最小化等价于原问题最大化

'''
复杂问题可以自定义类继承自GA类，然后重写其中的方法以实现更复杂的功能。
例如，可以重写选择、交叉或变异方法以引入自定义的遗传操作，或者重写适应度评估方法以实现多目标优化。
'''
# 2. 初始化GA（离散0-1变量，需指定precision=1）
ga = GA(
    func=knapsack_func,
    n_dim=5,                # 5个物品，维度=5
    size_pop=50,
    max_iter=200,
    prob_mut=0.05,
    lb=[0, 0, 0, 0, 0],    # 0-1变量，下界0
    ub=[1, 1, 1, 1, 1],    # 上界1
    precision=1            # 精度=1，强制变量为整数（0或1）
)

# 3. 运行算法
best_x, best_y = ga.run()

# 4. 解析结果
print("选中的物品（0=不选，1=选）：", best_x.astype(int))
print("总重量：", np.sum(best_x * weight))
print("最大总价值：", -best_y)  # 还原为原问题的最大值

plt.plot(ga.generation_best_Y)
plt.xlabel('epoch')
plt.ylabel('best_value')
plt.title('GA Knapsack Optimization Process')
plt.show()