import matplotlib.pyplot as plt
import numpy as np


x=np.linspace(0, 10, 100)
y=1/x

plt.grid(True)
plt.plot(x, y, color='c', marker='')
plt.show()


