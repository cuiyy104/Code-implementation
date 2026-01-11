import scipy
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def func_1(x, a, b):
    return a*x+b

x_data=np.linspace(0,10,10)
y_data=func_1(x_data,1,1)+np.random.normal(0,0.1,size=len(x_data))

'''三种插值方法'''
f_linear = interp1d(x_data, y_data, kind='linear')  # 线性插值
f_cubic = interp1d(x_data, y_data, kind='cubic')    # 三次插值
f_nearest = interp1d(x_data, y_data, kind='nearest')# 最近邻插值

x_new=np.linspace(0,10,100)

# 4. 计算插值结果
y_linear = f_linear(x_new)
y_cubic = f_cubic(x_new)
y_nearest = f_nearest(x_new)

# 5. 可视化对比
plt.scatter(x_data, y_data, label='true', color='black', s=20)
plt.plot(x_new, y_linear, label='linear', color='blue', linestyle='--')
plt.plot(x_new, y_cubic, label='cubic', color='red')
plt.plot(x_new, y_nearest, label='nearest', color='green', linestyle=':')
plt.legend()
plt.title('cmp')
plt.show()

