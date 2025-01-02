import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

data = np.loadtxt('data_5.text')
g=9.81
D=77/1000
m=1018/1000
l=94/100
L=l+D/2
t = data[:, 0]
U1 = data[:, 1]
U2 = data[:, 2]
L_1=70/100
k=3.4

t_smooth = np.linspace(t.min(), t.max(), 300)
spline_U1 = make_interp_spline(t, U1)
spline_U2 = make_interp_spline(t, U2)
U1_smooth = spline_U1(t_smooth)
U2_smooth = spline_U2(t_smooth)
omega_th = np.sqrt(g/L)
omega_0=np.sqrt(g/L)


omega_2 = np.sqrt(g/L)+k*L_1**2/(m*g**(1/2)*L**(3/2))

kappa = np.sqrt((omega_2**2-omega_0**2)/2)

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

finder = []
for i in range(0, len(U1)):
    if U2[i] >U2[0] and t[i]<25:
        finder.append(i)
print('omega_{n1}=', 2*np.pi*12/(t[finder[-1]]))
print('omega_{th}', omega_th)
print('omega_mean_1 -', (3.20+3.20+3.13+3.17+3.23)/5)
print('omega_mean_2 -', (3.07+ 3.20+ 3.13+ 3.21+ 3.20)/5)#3.07, 3.20, 3.13, 3.21, 3.20

