
import numpy as np
import matplotlib.pyplot as plt

# Заданные данные
U_B = np.array([7.5, 8.0, 8.5, 9.0])  # Напряжение
T = np.array([9.25, 8.69, 8.12, 7.69])  # Период колебаний (в секундах)
omega_exp = 2 * np.pi / T  # Угловая частота из периода
theta_0 = 0.3  # Амплитуда угловых колебаний

# Экспериментальные данные для амплитуды
A_0 = [2.8, 3.0, 3.8, 5.6, 5.9, 8.4, 17.4, 20.0, 20.0, 13.4, 7.2, 4.6, 4.2, 3.0, 2.8, 2.6, 2.4]
A_200 = [2.6, 3.0, 3.6, 4.2, 4.8, 6.0, 11.0, 11.8, 12.0, 6.2, 5.4, 4.2, 3.2, 3.0, 2.4, 2.2, 2.0]
A_400 = [1.8, 2.0, 2.2, 2.6, 2.9, 3.2, 3.5, 3.6, 3.4, 3.0, 2.8, 2.4, 2.2, 1.8, 1.7, 1.6, 1.4]

# Теоретическая частота для графика
omega_theory = np.linspace(0, 2 * omega_exp.mean(), 500)

# Коэффициенты затухания
betas = [0.1, 0.05, 0.01]

# График АЧХ
plt.figure(figsize=(12, 6))
for beta in betas:
    a_omega = (omega_exp.mean()**2 * theta_0) / np.sqrt((omega_exp.mean()**2 - omega_theory**2)**2 + 4 * beta**2 * omega_theory**2)
    plt.plot(omega_theory, a_omega, label=f'β = {beta}')

# Добавление экспериментальных точек
plt.scatter(omega_exp, A_0[:len(omega_exp)], color='blue', label='Экспериментальные данные t=0')
plt.scatter(omega_exp, A_200[:len(omega_exp)], color='orange', label='Экспериментальные данные t=200')
plt.scatter(omega_exp, A_400[:len(omega_exp)], color='green', label='Экспериментальные данные t=400')

# Настройка графика
plt.xlabel('ω, рад/с')
plt.ylabel('a(ω)')
plt.title('Амплитудно-частотная характеристика (АЧХ)')
plt.legend()
plt.grid()
plt.show()

