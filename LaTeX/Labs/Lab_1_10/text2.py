
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Данные из таблицы
U_B = np.array([7.0, 7.2, 7.5, 8.1, 8.2, 8.4, 8.5])  # Напряжение
T_exp = np.array([4.8, 4.2, 4.0, 3.2, 3.0, 2.4, 2.2])  # Период колебаний (в секундах)
a_exp_0 = np.array([3.0, 3.8, 4.6, 7.0, 6.2, 3.0, 2.6])  # Амплитуда при t=0
a_exp_200 = np.array([2.6, 3.6, 4.2, 6.2, 4.2, 2.4, 2.2])  # Амплитуда при t=200
a_exp_400 = np.array([1.8, 2.2, 2.0, 3.4, 2.2, 1.8, 1.4])  # Амплитуда при t=400

# Расчёт экспериментальных угловых частот
omega_exp = 2 * np.pi / T_exp

# Аппроксимация для градуировочного графика
def omega_U_func(U, a, b):
    return a * U + b

params, _ = curve_fit(omega_U_func, U_B, omega_exp)
omega_U_fit = omega_U_func(U_B, *params)

# Построение градуировочного графика
plt.figure(figsize=(10, 6))
plt.plot(U_B, omega_exp, 'o', label='Экспериментальные данные')
plt.plot(U_B, omega_U_fit, '-', label=f'Аппроксимация: $\\omega(U) = {params[0]:.2f} U + {params[1]:.2f}$')
plt.xlabel('U_B, В')
plt.ylabel('ω, рад/с')
plt.title('Градуировочный график зависимости ω от U')
plt.legend()
plt.grid()
plt.show()

# Заданные параметры для теоретических графиков АЧХ и ФЧХ
omega_0 = params[0] * 8.0 + params[1]  # Оценка собственной частоты по аппроксимации
theta_0 = 1.0
betas = [0.009869271456738608, 0.05806619917546153, 0.19701454928271458]
omega = np.linspace(0, 2 * omega_0, 500)

# Построение АЧХ с экспериментальными точками
plt.figure(figsize=(12, 6))
for beta in betas:
    a_omega = (omega_0**2 * theta_0) / np.sqrt((omega_0**2 - omega**2)**2 + 4 * beta**2 * omega**2)
    plt.plot(omega, a_omega, label=f'β = {beta}')
plt.plot(omega_exp, a_exp_0, 'o', label='Экспериментальные точки t=0')
plt.plot(omega_exp, a_exp_200, 'o', label='Экспериментальные точки t=200')
plt.plot(omega_exp, a_exp_400, 'o', label='Экспериментальные точки t=400')
plt.xlabel('ω, рад/с')
plt.ylabel('a(ω)')
plt.title('Амплитудно-частотная характеристика (АЧХ)')
plt.legend()
plt.grid()
plt.show()

# Построение ФЧХ с экспериментальными точками
plt.figure(figsize=(12, 6))
for beta in betas:
    delta = -np.arctan2(2 * beta * omega, omega_0**2 - omega**2)
    plt.plot(omega, delta, label=f'β = {beta}')
plt.xlabel('ω, рад/с')
plt.ylabel('δ(ω), рад')
plt.title('Фазово-частотная характеристика (ФЧХ)')
plt.legend()
plt.grid()
plt.show()

