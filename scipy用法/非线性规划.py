from scipy.optimize import minimize
import numpy as np

def func(x):
    return x[0]**2 + x[1]**2 + np.sin(x[0]+x[1])

cst=[# 非线性不等式约束：x0² + x1² - 4 ≤ 0 → 转换为 4 - x0² - x1² ≥ 0
    {'type': 'ineq', 'fun': lambda x: 4 - (x[0]**2 + x[1]**2)},
    # 线性等式约束：x0 + x1 - 1 = 0
    {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1}
]
bounds=[(0,np.inf),(-np.inf,np.inf)]
x0=[0, 0]

result=minimize(func, x0, method='SLSQP',bounds=bounds,constraints=cst,tol=1e-6,options={'disp':True})

print(f"\n最优解:{result.x}, 最优值: {result.fun}, 是否成功: {result.success}")
