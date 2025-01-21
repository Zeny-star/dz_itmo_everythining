import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

t = np.array([18.4, 18.38, 18.38, 18.41])
phi_0 = np.pi / 6 
l_1=57/1000
l_0=25/1000
b=40/1000
m=405/1000
I_0=0.008
g=9.81
Delta_m = 0.0001
Delta_R = 0.0001

T = np.mean(t) / 10
print(T)
t_1 = np.array([37.02, 36.98, 37.38, 37.48])
t_2 = np.array([79.95, 81.24, 81.28, 81.26])
t_3 = np.array([130.81, 133.1, 132.02, 131.39])
t_4 = np.array([193.67, 195.41, 193.3, 193.59])
t_5 = np.array([270.72, 272.67, 271.75, 270.29])
amplitude = np.array([25, 20, 15, 10, 5])
amplitude = amplitude * np.pi / 180 
A_0 = np.pi / 6  
time = np.array([np.mean(t_1), np.mean(t_2), np.mean(t_3), np.mean(t_4), np.mean(t_5)])

plt.figure(figsize=(12, 6))
plt.plot(time, amplitude, 'o', label='Амплитуда A(t)', color='blue')

coeffs_linear = np.polyfit(time, amplitude, 1)
linear_fit = np.polyval(coeffs_linear, time)


plt.plot(time, linear_fit, '--', label=f'Линейная аппроксимация\nk={coeffs_linear[0]:.4f}', color='red')
plt.xlabel('Время t (с)')
plt.ylabel('Амплитуда A (рад)')
plt.title('Зависимость амплитуды от времени A(t)')
plt.grid(True)
plt.legend()

delta_phi = -coeffs_linear[0] / (4 * T)
print('Коэффициенты линейной аппроксимации:', coeffs_linear[0])
print(f"Ширина зоны застоя ∆φз = {delta_phi:.4f} рад")

n_stop = A_0 / (4 * delta_phi)  
print(f"Число периодов до прекращения колебаний: n ≈ {n_stop:.2f}")

plt.show()



t_1_2 = np.array([16.44, 16.47, 16.45, 16.47])
t_2_2 = np.array([17.38, 17.44, 17.37, 17.38])
t_3_2 = np.array([18.38, 18.40, 18.38, 18.41])
t_4_2 = np.array([19.81, 19.71, 19.75, 19.72])
t_5_2 = np.array([21.07,  20.87, 21.19, 21.19])
t_6_2 = np.array([22.72, 22.75, 22.72, 22.87])

T_mean = np.array([
    np.mean(t_1_2),
    np.mean(t_2_2),
    np.mean(t_3_2),
    np.mean(t_4_2),
    np.mean(t_5_2),
    np.mean(t_6_2),
]) / 10  # Приводим к периоду одного цикла

# Квадраты периодов
T_squared = T_mean**2

# Геометрия маятника
l_1 = 57 / 1000  # Верхний радиус
l_0 = 25 / 1000  # Длина сегмента
b = 40 / 1000  # Высота
m = 405 / 1000  # Масса
I_0 = 0.008  # Начальный момент инерции

# Вычисление моментов инерции
R_up = l_1 + b / 2
R_down = l_1 + 5 * l_0 + b / 2
R_side = [l_1 + (n - 1) * l_0 + b / 2 for n in range(1, 7)]
R_side = np.array(R_side)


I_gr = m * (R_up**2 + R_down**2 + 2 * R_side**2)
I = I_gr + I_0  # Полный момент инерции

print(f'I = {I}')
# Построение графика T^2(I)
plt.figure(figsize=(12, 6))
plt.plot(I, T_squared, 'o', label='Данные $T^2(I)$', color='blue')

# Линейная аппроксимация
coeffs = np.polyfit(I, T_squared, 1)
linear_fit = np.polyval(coeffs, I)

# Отображение линейной аппроксимации
plt.plot(I, linear_fit, '--', label=f'Линейная аппроксимация\nk={coeffs[0]:.4f}', color='red')
plt.xlabel('Момент инерции I (кг·м²)')
plt.ylabel('$T^2$ (с²)')
plt.title('Зависимость $T^2(I)$')
plt.grid(True)
plt.legend()

# Вывод углового коэффициента
print('Коэффициенты линейной аппроксимации:', coeffs)
print(f"Угловой коэффициент k = {coeffs[0]:.4} с²/(кг·м²)")
plt.show()

l_th = (R_down-R_up)/4

l_pr_th = I/(4*m*l_th)
l_pr_ex = g*T_mean**2/(4*np.pi**2)


print('ml_ex', 4*np.pi**2/(g*coeffs[0]))
print('l_th', l_th)
print('l_pr_th', l_pr_th)
print('l_pr_ex', l_pr_ex)

Delta_l_th = np.sqrt(Delta_R**2 + Delta_R**2) / 4

Delta_I_up = np.sqrt((2 * m * R_up * Delta_R)**2 + (R_up**2 * Delta_m)**2)
Delta_I_down = np.sqrt((2 * m * R_down * Delta_R)**2 + (R_down**2 * Delta_m)**2)
Delta_I_side = np.sqrt(np.sum((2 * m * R_side * Delta_R)**2 + (R_side**2 * Delta_m)**2) / len(R_side))
Delta_I = np.sqrt(Delta_I_up**2 + Delta_I_down**2 + Delta_I_side**2)


# Теоретическая приведённая длина
l_pr_th = I / (4 * m * l_th)

# Погрешность теоретической приведённой длины
Delta_l_pr_th = np.sqrt(
    (Delta_I / (4 * m * l_th))**2 +
    (I * Delta_m / (4 * m**2 * l_th))**2 +
    (I * Delta_l_th / (4 * m * l_th**2))**2
)

# Вывод результатов
print(f"Теоретическая приведённая длина: l_pr_th = {l_pr_th.mean():.6f} м")
print(f"Погрешность теоретической приведённой длины: Delta_l_pr_th = {Delta_l_pr_th.mean():.6f} м")

