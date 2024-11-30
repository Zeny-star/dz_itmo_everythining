import matplotlib.pyplot as plt
import numpy as np


x=[10, 13, 15, 20, 22, 25]
y=[20, 23, 29, 25, 23, 29]

plt.grid()
plt.plot(x, y, color='g', marker='.')
plt.bar(x, y)
plt.show()
