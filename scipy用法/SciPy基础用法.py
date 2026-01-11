import scipy
import numpy as np
from scipy.integrate import quad
#数值积分和常微分方程求解器
from scipy import optimize
#函数最小化，曲线拟合和求根，线性规划
from scipy import linalg
###积分计算
def hanshu(x):
    return x**2

ans,err=quad(hanshu,0,1)
print("积分结果为:",ans)
print("误差为:",err)

A=np.array([[1,2],[3,4]])
#计算矩阵的拟
inv_A=linalg.inv(A)
print("矩阵的逆:",inv_A)

b=np.array([1,2])

x=np.dot(inv_A,b)
print("A的拟乘上b的结果:",x)

ans2=optimize.minimize(hanshu,x0=1,method='BFGS')
print("最小值:",ans2)
'''result = minimize(fun, x0, args=(), method=None, bounds=None, 
constraints=(), tol=None, options=None)'''
'''这个函数的参数分别是：
fun: 要最小化的目标函数。
x0: 初始猜测值。
args: 传递给目标函数的额外参数。
method: 优化算法的方法。
bounds: 变量的边界条件。
constraints: 约束条件。
tol: 容差，用于停止准则。
options: 其他选项，如最大迭代次数等。'''

