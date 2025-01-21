import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

t = [0, 0.0333, 0.0667, 0.100, 0.133, 0.167, 0.200, 0.233, 0.267, 0.300, 0.333, 0.367, 0.400, 0.433] #секунды

x = [-0.015432300, 0.037146270, 0.087998310, 0.141616300, 0.192701800, 0.243317800, 0.295282500, 0.346972300, 0.396763800, 0.446280500, 0.494727700, 0.545906700, 0.594835800, 0.644999100] #метры

y = [0.2737373, 0.3277853, 0.3757058, 0.4127315, 0.4424951, 0.4564396, 0.4637793, 0.4618590, 0.4480238, 0.4249286, 0.3918248, 0.3466686, 0.2907664, 0.2244010] #метры

teta = [44.592170, 39.115080, 32.533370, 23.256540, 11.721840, 2.992863, -8.825016, -20.398980, -29.841600, -38.150920, -45.270650, -50.976770, -55.299930,
-59.453300] #угол в обычных градусах

g = 9.81

t_fine = np.linspace(np.array(t).min(), np.array(t).max(), 500)  # новые точки времени для гладкости
x_interp = interp1d(t, x, kind='cubic')(t_fine)
y_interp = interp1d(t, y, kind='cubic')(t_fine)
fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

axs[0].plot(t, x, 'o-', label="$x(t)$ (основной)", color="#5986e4", lw=1.5, markersize=4)
axs[0].plot(t_fine, x_interp, label="$x(t)$ (аппроксимация)", color="#b01bb3", lw=1.2, linestyle="--")
axs[0].set_ylabel("x (м)", fontsize=14)
axs[0].set_title("Зависимость $x(t)$ и $y(t)$", fontsize=16)
axs[0].legend(fontsize=12)

axs[1].plot(t, y, 'o-', label="$y(t)$ (основной)", color="#5986e4", lw=1.5, markersize=4)
axs[1].plot(t_fine, y_interp, label="$y(t)$ (аппроксимация)", color="#b01bb3", lw=1.2, linestyle="--")
axs[1].set_ylabel("y (м)", fontsize=14)
axs[1].set_xlabel("t (с)", fontsize=14)
axs[1].legend(fontsize=12)

plt.tight_layout()
plt.show()


t_up_theory = t_fine[list(y_interp).index(y_interp.max())]
for i in range(50, len(y_interp)):
    if y_interp[i] < y_interp[0]:
        t_down_theory = t_fine[i]
        x_down_theory = x_interp[i]-x_interp[1]
        break



vx_exp_0=(x[2]-x[1])/(t[2]-t[1])
vy_exp_0=(y[2]-y[1])/(t[2]-t[1])
vx_exp_1=(x[0]-x[1])/(t[0]-t[1])
vy_exp_1=(y[0]-y[1])/(t[0]-t[1])
vx_exp_2=(x[2]-x[0])/(t[2]-t[0])
vy_exp_2=(y[2]-y[0])/(t[2]-t[0])

v_0 = np.sqrt(vx_exp_0**2+vy_exp_0**2)
v_1 = np.sqrt(vx_exp_1**2+vy_exp_1**2)
v_2 = np.sqrt(vx_exp_2**2+vy_exp_2**2)


t_up_0 = v_0*np.sin(teta[1])/g
t_up_1 = v_1*np.sin(teta[1])/g
t_up_2 = v_2*np.sin(teta[1])/g

t_down_0 = t_up_0*2
t_down_1 = t_up_1*2
t_down_2 = t_up_2*2

x_0 = vx_exp_0*t_down_0
x_1 = vx_exp_1*t_down_1
x_2 = vx_exp_2*t_down_2

print(t_up_theory, t_up_0, (1-t_up_theory/t_up_0)*100)
print(t_down_theory,t_down_0, (1-t_down_theory/t_down_0)*100)
print(x_down_theory,x_0, (1-x_down_theory/x_0)*100)


print(t_up_theory, t_up_1, (1-t_up_theory/t_up_1)*100)
print(t_down_theory,t_down_1, (1-t_down_theory/t_down_1)*100)
print(x_down_theory,x_1, (1-x_down_theory/x_1)*100)

print(t_up_theory, t_up_2, (1-t_up_theory/t_up_2)*100)
print(t_down_theory,t_down_2, (1-t_down_theory/t_down_2)*100)
print(x_down_theory,x_2, (1-x_down_theory/x_2)*100)


