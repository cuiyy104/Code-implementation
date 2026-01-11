import scipy
import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt


def func(x,y):
    return np.sin(np.sqrt(x**2+y**2))

x_data=np.linspace(0,10,10)
y_data=np.linspace(0,10,10)
X,Y=np.meshgrid(x_data,y_data)
z=func(X,Y)
f=interp2d(x_data,y_data,z,kind='cubic')

x_new=np.linspace(0,10,100)
y_new=np.linspace(0,10,100)
z_new=f(x_new,y_new)
X_new,Y_new=np.meshgrid(x_new,y_new)

fig=plt.figure(figsize=(12,5))

ax1=fig.add_subplot(121,projection='3d')
ax1.plot_surface(X,Y,z,cmap='viridis')
ax1.set_title('Original Data')

ax2=fig.add_subplot(122,projection='3d')
ax2.plot_surface(X_new,Y_new,z_new,cmap='viridis')
ax2.set_title('Interpolated Data')

plt.show()