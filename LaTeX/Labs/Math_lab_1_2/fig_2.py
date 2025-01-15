import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

t = [0.000000,
0.033333,
0.066667,
0.100000,
0.133333,
0.166667,
0.200000,
0.233333,
0.266667,
0.300000,
0.333333,
0.366667,
]

x = [-0.03933419,
0.04581760,
0.13141310,
0.23007100,
0.31393630,
0.40529850,
0.49829200,
0.59408320,
0.69159150,
0.78148900,
0.87966370,
0.98386900]

y = [0.69763640,
0.71607950,
0.72510890,
0.71133410,
0.68642850,
0.64611500,
0.58767510,
0.50167390,
0.40043640,
0.28640860,
0.13453320,
-0.03342968]
O = [
9.140305,
-1.475312,
-11.965140,
-20.415120,
-28.176500,
-37.419950,
-44.087560,
-48.957800,
-54.728380,
-57.676120,
]
#plt.plot(t, y)
#plt.show()
t_fine = np.linspace(np.array(t).min(), np.array(t).max(), 500)  # новые точки времени для гладкости
x_interp = interp1d(t, x, kind='cubic')(t_fine)  # аппроксимация x(t)
y_interp = interp1d(t, y, kind='cubic')(t_fine)  # аппроксимация y(t)
fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

# Основной график x(t)
axs[0].plot(t, x, 'o-', label="$x(t)$ (основной)", color="#5986e4", lw=1.5, markersize=4)
axs[0].plot(t_fine, x_interp, label="$x(t)$ (аппроксимация)", color="#b01bb3", lw=1.2, linestyle="--")
axs[0].set_ylabel("x (м)", fontsize=14)
axs[0].set_title("Зависимость $x(t)$ и $y(t)$", fontsize=16)
axs[0].legend(fontsize=12)

# Основной график y(t)
axs[1].plot(t, y, 'o-', label="$y(t)$ (основной)", color="#5986e4", lw=1.5, markersize=4)
axs[1].plot(t_fine, y_interp, label="$y(t)$ (аппроксимация)", color="#b01bb3", lw=1.2, linestyle="--")
axs[1].set_ylabel("y (м)", fontsize=14)
axs[1].set_xlabel("t (с)", fontsize=14)
axs[1].legend(fontsize=12)

# Настройка внешнего вида
plt.tight_layout()
plt.show()


#подсчет производных
vx_exp_0=(x[1]-x[0])/(t[1]-t[0])
vy_exp_0=(y[1]-y[0])/(t[1]-t[0])
vx_exp_1=(x[0]-x[1])/(t[0]-t[1])
ky_exp_1=(y[0]-y[1])/(t[0]-t[1])
vx_exp_2=(x[2]-x[0])/(2*(t[2]-t[1]))
vy_exp_2=(y[2]-y[0])/(2*(t[2]-t[1]))


print(t[vy.index(np.array(vy).max())])
