import numpy as np
from scipy import optimize
from scipy.optimize import milp

sizes = np.array([21, 11, 15, 9, 34, 25, 41, 52])
values = np.array([22, 12, 16, 10, 35, 26, 42, 53])

bounds=optimize.Bounds(0,1)
is_choice=np.full_like(sizes,True)

wei=100
cst=optimize.LinearConstraint(A=sizes,lb=0,ub=wei)
res = milp(c=-values, constraints=cst,
           integrality=is_choice, bounds=bounds)

print(f"选择情况：{res.x}，总价值：{-res.fun}, 是否成功：{res.success}")