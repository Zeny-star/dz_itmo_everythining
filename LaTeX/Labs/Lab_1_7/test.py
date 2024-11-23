import numpy as np


h1=[32.0/100, 31.1/100, 31.5/100, 31.5/100, 31.2/100]
t=[1.241, 1.250, 1.254, 1.252, 1.24, 1.237]
h=34.7/100

d=9.9/1000
D=86.2/1000
delta=0.07/1000#для всего кроме таблицы
delta_grigoria=1/1000 #для таблицы
l=189.4/1000
m=159.5/1000
x=8.24/1000
g=9.81
r=d/2
R=D/2

h1_mean = np.mean(np.array(h1))
t_mean = np.mean(np.array(t))
I_1=m*r**2*(g*t_mean**2/(2*h)-1)
I_2=m*r**2*(g*t_mean**2*h1_mean/(h*(h+h1_mean))-1)
delta_m = 0.1 / 1000
delta_t = np.std(t) / np.sqrt(len(t))
delta_h = delta_grigoria
delta_h1 = np.std(h1) / np.sqrt(len(h1))

A = g * t_mean**2 * h1_mean / (h * (h + h1_mean)) - 1
delta_A = A * np.sqrt((2 * delta_t / t_mean)**2 + (delta_h / h)**2 + (delta_h1 / h1_mean)**2)
delta_I = I_2 * np.sqrt((delta_m / m)**2 + (2 * delta / r)**2 + (delta_A / A)**2)
m_diska=m*(x*np.pi*R**2/(l*np.pi*r**2+x*np.pi*R**2))
I_theory=m_diska*R**2/2
print(f"Среднее время t_mean: {t_mean:.4f} с")
print(f"Средняя высота h1_mean: {h1_mean:.4f} м")
print(f"Момент инерции I_1 по формуле (6): {I_1} кг·м²")
print(f"Момент инерции I_2 по формуле (11): {I_2} кг·м²")
print(f"Момент инерции I_theory по формуле: {I_theory} кг·м²")
print(f"Погрешность для I_2: {delta_I} кг·м²")

print(m, m_diska)
