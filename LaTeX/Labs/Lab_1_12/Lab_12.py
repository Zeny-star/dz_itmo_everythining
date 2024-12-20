import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

data = np.loadtxt('data1.1.txt')
t = data[:, 0]
U1 = data[:, 1]
U2 = data[:, 2]

t_smooth = np.linspace(t.min(), t.max(), 300)
spline_U1 = make_interp_spline(t, U1)
spline_U2 = make_interp_spline(t, U2)
U1_smooth = spline_U1(t_smooth)
U2_smooth = spline_U2(t_smooth)

plt.figure(figsize=(10, 5))
plt.plot(t_smooth, U1_smooth, label='U1(t)', color='red')
plt.plot(t_smooth, U2_smooth, label='U2(t)', color='orange')
plt.xlabel('Время t [с]')
plt.ylabel('Напряжение U [В]')
plt.title('Зависимость напряжений U1 и U2 от времени t')
plt.xlim(left=0)
plt.legend(loc='upper right')
plt.grid()
plt.show()
