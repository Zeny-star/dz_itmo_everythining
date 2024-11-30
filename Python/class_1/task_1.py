import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x=np.linspace(-10, 0, 1000)
x_add=np.linspace(0, 10, 1000)
y=-3*x+10
y_1=1/x
y_1_add=1/x_add
some_data = np.random.randn(1000)
x_stepped=np.arange(-20, -1, 0.5)
y_stepped=x_stepped/2


p = norm.pdf(x, np.mean(some_data), np.std(some_data))


x_1=np.linspace(min(some_data), max(some_data), 1000)
fig, axs = plt.subplots(2, 2)
axs[0, 0].grid()
axs[0, 1].grid()
axs[1, 0].grid()
axs[1, 1].grid()
axs[0, 0].plot(x, y, color='c')
axs[0, 1].plot(x, y_1, color='g')
axs[0, 1].plot(x_add, y_1_add, color='g')
axs[1, 0].plot(x_1, p,  color='b')
axs[1, 1].plot(x_stepped, y_stepped, color='r')
plt.show()
