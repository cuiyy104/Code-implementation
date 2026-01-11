'''处理标准最小化问题
如果是最大化问题，可以将目标函数取负号转化为最小化问题'''
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt

'''# 求解线性规划
result = linprog(
    c,          # 目标函数系数
    A_ub=None,  # 不等式约束矩阵 A
    b_ub=None,  # 不等式约束右端 b
    A_eq=None,  # 等式约束矩阵 A_eq
    b_eq=None,  # 等式约束右端 b_eq
    bounds=None,# 变量边界
    method='highs'  # 求解算法（推荐highs，高效稳定）
)'''

c=[3,4]
# 不等式约束：x0+2x1 ≥5 → 转换为 -x0-2x1 ≤ -5
A_ub=[[-1,-2]]
b_ub=[-5]

# 等式约束：x0 + x1 = 3
A_eq = [[1, 1]]
b_eq = [3]

# 变量边界
bounds = [(0, np.inf), (0, np.inf)]

result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
print(f"最优解: {result.x}, 最优值: {result.fun}, 是否成功: {result.success}")