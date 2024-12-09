import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
m_0 = 24.08 / 1000  # масса держателя, кг
m_1 = 44.05 / 1000  # масса одного груза, кг
m_2 = 64.02 / 1000  # масса второго груза, кг
r = 12.5 / 100  # радиус маховика, м
l = 22.5 / 100  # плечо силы, м
delta_l = 1 / 10000  # погрешность плеча, м
g = 9.81  # ускорение свободного падения, м/с²
delta_m = 0.01 / 1000  # погрешность массы, кг
m_m=1.5

# Данные измерений
mu_1_1 = np.array([393.7, 692.4, 479.7, 396.2, 311.3])
mu_1_2 = np.array([349.8, 550.7, 418.3, 336.1, 281.3])
t_1 = np.array([55.85, 90.15, 65.35, 52.25, 44.25])

mu_2_1 = np.array([496.0, 385.1, 318.4, 518.7, 449.3])
mu_2_2 = np.array([446.2, 359.7, 300.0, 478.2, 417.2])
t_2 = np.array([35.12, 27.91, 22.81, 36.44, 32.19])

mu_3_1 = np.array([614.5, 652.6, 565.4, 468.9, 580.9])
mu_3_2 = np.array([562.5, 592.5, 510.2, 442.5, 540.2])
t_3 = np.array([27.54, 31.85, 27.21, 23.08, 28.14])

# Массы грузов
masses = np.array([m_0, m_0+m_1, m_0+m_2])

# Вычисление средней угловой скорости
def calculate_omega(mu1, mu2):
    return 2 * np.pi * (mu1 + mu2) / (2 * 60)

omega_1 = calculate_omega(mu_1_1, mu_1_2)
omega_2 = calculate_omega(mu_2_1, mu_2_2)
omega_3 = calculate_omega(mu_3_1, mu_3_2)

# Теоретический момент инерции
def calculate_theoretical_inertia(mass, radius):
    return mass * radius**2 / 2

I_theor = calculate_theoretical_inertia(m_m, r)

# Обработка результатов по МНК
def calculate_A_and_sigma(t, omega):
    A = np.sum(omega * t) / np.sum(omega**2)
    sigma_A = np.sqrt(np.sum((t - A * omega)**2) / (len(t) - 1) / np.sum(omega**2))
    return A, sigma_A

A_1, sigma_A1 = calculate_A_and_sigma(t_1, omega_1)
A_2, sigma_A2 = calculate_A_and_sigma(t_2, omega_2)
A_3, sigma_A3 = calculate_A_and_sigma(t_3, omega_3)

# Экспериментальный момент инерции
def calculate_experimental_inertia(A, mass, g, l):
    return A * mass * g * l / (2 * np.pi)

I_exp_1 = calculate_experimental_inertia(A_1, masses[0], g, l)
I_exp_2 = calculate_experimental_inertia(A_2, masses[1], g, l)
I_exp_3 = calculate_experimental_inertia(A_3, masses[2], g, l)

# Построение графиков
def plot_graph_with_error(t, omega, A, sigma_A, label):
    plt.errorbar(
        omega, 
        t, 
        yerr=2 * sigma_A,  # Доверительный интервал (2σ)
        fmt='o', 
        label=f'Эксперимент {label}',
        capsize=5, 
        capthick=1
    )
    plt.plot(omega, A * omega, linestyle='--', label=f'Линейная зависимость {label}')

plt.figure(figsize=(10, 6))
plot_graph_with_error(t_1, omega_1, A_1, sigma_A1, "1")
plot_graph_with_error(t_2, omega_2, A_2, sigma_A2, "2")
plot_graph_with_error(t_3, omega_3, A_3, sigma_A3, "3")
plt.xlabel("Средняя угловая скорость ω (рад/с)")
plt.ylabel("Период прецессии T (с)")
plt.title("Зависимость периода прецессии от угловой скорости с учётом погрешностей")
plt.legend()
plt.grid()
plt.show()

# Вывод результатов
print(f"Теоретический момент инерции I_theor: {I_theor:.5f} кг·м²")
print(f"Экспериментальные моменты инерции:")
print(f"  I_exp_1: {I_exp_1:.5f} ± {sigma_A1/2:.5f} кг·м²")
print(f"  I_exp_2: {I_exp_2:.5f} ± {sigma_A2:.5f} кг·м²")
print(f"  I_exp_3: {I_exp_3:.5f} ± {sigma_A3:.5f} кг·м²")
