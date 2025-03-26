import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

colors = ['#5986e4', '#7d4dc8', '#b01bb3']

p_0 = 10**5
t_1 = 21.5+273.15
p_1_1 = np.array([35.6, 14.6, 2.9, -9.2, -27.7, -38.9, -48.1, -52.1])*13.55*9.819
p_1_2 = np.array([35.6, 0.3, -13, -25.2, -33.7, -41.5, -47.3, -52.1])*13.55*9.819

t_2 = 31.2+273.15
p_2_1 = np.array([18.2, 1.9, -12.1, -23.3, -32, -39.6, -51.3, -55.1])*13.55*9.819
p_2_2 = np.array([12.9, -6.1, -18.6, -28.9, -37.4, -44.9, -50.2, -55.1])*13.55*9.819

t_3 = 40.1+273.15
p_3_1 = np.array([12.6, -4.2, -15.9, -27.9, -35.5, -42.5, -49.3, -53.2])
p_3_2 = np.array([12.5, -1.3, -15.2, -27, -35.4, -42.8, -48.7, -53.2])


t_4 = 49+273.15
p_4_1 = np.array([15.4, -1.7, -14.7, -25.4, -34.8, -41.4, -50.8, -55.3])
p_4_2 = np.array([11.7, -3.9, -18.5, -28.6, -37.2, -44.7, -50, -55.4])

t_5 = 60+273.15
p_5_1 = np.array([13.6, -4.5, -15.5, -27.6, -36.2, -42.4, -49, -53.3])
p_5_2 = np.array([13.6, -0.2, -16.7, -26.5, -35.5, -42.4, -48.1, -53.3])

v = np.array([50, 60, 70, 80, 90, 100, 110, 120])
v = v * 1e-6

p_1 = p_0 + (p_2_1+p_2_2)/2
p_2 = p_0 + (p_2_1+p_2_2)/2
p_3 = p_0 + (p_3_1+p_3_2)/2
p_4 = p_0 + (p_4_1+p_4_2)/2
p_5 = p_0 + (p_5_1+p_5_2)/2
y_1 = 1/p_1
y_2 = 1/p_2
y_3 = 1/p_3
y_4 = 1/p_4
y_5 = 1/p_5

A = np.vstack([v, np.ones(len(v))]).T  
a_1, b_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]  
a_2, b_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
a_3, b_3 = np.linalg.lstsq(A, y_3, rcond=None)[0]
a_4, b_4 = np.linalg.lstsq(A, y_4, rcond=None)[0]
a_5, b_5 = np.linalg.lstsq(A, y_5, rcond=None)[0]

plt.plot(y_4, v, 'bo', label='Экспериментальные точки', color = colors[0])
plt.plot(a_4*v + b_4,v, 'r-', label=f'МНК', color = colors[2])
plt.ylabel('Объем $v$ (м³)')
plt.xlabel('$1/p$ (1/Па)')
plt.legend()
plt.show()

t = np.array([t_1, t_2, t_3, t_4, t_5])
a = np.array([a_1, a_2, a_3, a_4, a_5])
slope, intercept, r_value, p_value, std_err = stats.linregress(t, a)

A = slope
C = intercept

t_abs_zero = -C / A

n = len(t)
x_mean = np.mean(t)
S_xx = np.sum((t - x_mean)**2)
delta_A = std_err * np.sqrt(n / S_xx)
delta_C = std_err * np.sqrt(1/n + x_mean**2/S_xx)

delta_t_abs_zero = t_abs_zero * np.sqrt((delta_A/A)**2 + (delta_C/C)**2)
print(t_abs_zero, delta_t_abs_zero)

plt.plot(t, a)
plt.ylabel('a')
plt.xlabel('t')
plt.show()
