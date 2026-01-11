import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from sympy.codegen.ast import integer

groups = [
    [(2, 3), (3, 4)],  # 第1组
    [(1, 2), (4, 5)],  # 第2组
    [(5, 6)]            # 第3组
]

C=8

wei=[]
val=[]
group_idx=[]
'''细节就是分组逐个增加限制条件'''
for id,items in enumerate(groups):
    for item in items:
        wei.append(item[0])
        val.append(item[1])
        group_idx.append(id)

n_len=len(wei)

c=-np.array(val)

cst=[]
'''第一个约束'''
unique_grps=np.unique(group_idx)
for g in unique_grps:
    group_var_idx = [i for i, gid in enumerate(group_idx) if gid == g]
    coeffs=np.zeros(n_len)
    coeffs[group_var_idx]=1
    cst.append(LinearConstraint(coeffs,lb=0,ub=1))
'''第二个约束'''
cst.append(LinearConstraint(wei,lb=0,ub=C))
bounds=Bounds(lb=np.zeros(n_len),ub=np.ones(n_len))

ans=milp(c=c,constraints=cst,bounds=bounds,options={'disp':True})
print(f"选择情况：{ans.x}，总价值：{-ans.fun}, 是否成功：{ans.success}")