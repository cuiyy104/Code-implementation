# python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def hanshu(x, a, b):
    return a * x + b
#定义残差函数
def residuals(params, x, y):
    a, b = params
    return hanshu(x, a, b) - y  # 模型 - 观测（也可用 y - 模型）

# 生成模拟数据
x_data = np.linspace(0, 10, 100)
true_a, true_b = 2, 5
y_true = hanshu(x_data, true_a, true_b)
y_data = y_true + np.random.normal(0, 1, size=len(x_data))

# 初始猜测并拟合
x0 = [1.0, 1.0]
ans = optimize.least_squares(residuals, x0, args=(x_data, y_data))
fitted_a, fitted_b = ans.x
print("拟合结果: a =", fitted_a, "b =", fitted_b, "是否收敛:", ans.success)
#计算残差向量
x=residuals(ans.x, x_data, y_data)
print(x,x.shape)
#可视化
plt.scatter(x_data, y_data,s=10, label='data')
plt.plot(x_data,y_true,'r-',label='true')
plt.plot(x_data,hanshu(x_data,fitted_a,fitted_b),'g-',label='fit')
plt.legend()
plt.show()