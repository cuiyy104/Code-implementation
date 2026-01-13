# python
import numpy as np
import inspect
from sko.DE import DE

# 查看 DE.__init__ 的签名，确认可用参数名
print(inspect.signature(DE.__init__))
print(DE.__doc__ or "")

# 最小示例：目标函数和运行
def obj_func(x):
    return np.sum(x**2)

D = 3 # 维度
N = 20 # 种群规模
G = 100 # 最大迭代次数
lb = [-5.0] * D # 下界
ub = [5.0] * D # 上界

de = DE(func=obj_func, n_dim=D, size_pop=N, max_iter=G, lb=lb, ub=ub)

# 运行并输出结果
best_x, best_y = de.run()
print("best_x:", best_x, "best_y:", best_y)
