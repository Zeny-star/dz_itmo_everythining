import numpy as np
import matplotlib.pyplot as plt

h1 = np.array([202, 202, 202, 202, 202,
               213, 213, 213, 213, 213,
               222, 222, 222, 222, 222,
               231, 231, 231, 231, 231,
               242, 242, 242, 242, 242])  # мм

h2 = np.array([192, 192, 192, 192, 192,
               193, 193, 193, 193, 193,
               194, 194, 194, 194, 194,
               195, 195, 195, 195, 195,
               195, 195, 195, 195, 195])  # мм

t1 = np.array([1.4, 1.3, 1.3, 1.3, 1.3,
               0.9, 0.9, 0.9, 0.9, 0.9,
               0.7, 0.7, 0.7, 0.7, 0.7,
               0.6, 0.6, 0.6, 0.6, 0.6,
               0.6, 0.5, 0.5, 0.6, 0.5])

t2 = np.array([4.5, 4.4, 4.4, 4.4, 4.4,
               3.0, 3.0, 3.0, 3.0, 3.0,
               2.5, 2.5, 2.5, 2.5, 2.5,
               2.1, 2.1, 2.1, 2.1, 2.1,
               1.9, 1.9, 1.9, 1.9, 1.9])

h1 = h1 / 1000  # м
h2 = h2 / 1000  # м

x1 = 0.15  # м
x2 = 1.10  # м
delta_x1 = 0.005  # м
delta_x2 = 0.005  # м

h0 = 192 / 1000  # м
h0_1 = 192 / 1000  # м

sin_alpha = ((h1 - h2) - (h0 - h0_1)) / ((1 - 0.22))

acceleration = 2 * (x2 - x1) / (t2**2 - t1**2)

mean_t1 = np.mean(t1)
mean_t2 = np.mean(t2)
std_t1 = np.std(t1, ddof=1) / np.sqrt(len(t1))
std_t2 = np.std(t2, ddof=1) / np.sqrt(len(t2))

delta_a = acceleration * np.sqrt(
    ((delta_x2**2 + delta_x1**2) / (x2 - x1)**2) +
    (4 * ((mean_t1 * std_t1)**2 + (mean_t2 * std_t2)**2) / (mean_t2**2 - mean_t1**2)**2)
)

N = len(sin_alpha)
sum_a = np.sum(acceleration)
sum_sin_alpha = np.sum(sin_alpha)
sum_a_sin_alpha = np.sum(acceleration * sin_alpha)
sum_sin_alpha_sq = np.sum(sin_alpha**2)

B = (N * sum_a_sin_alpha - sum_a * sum_sin_alpha) / (N * sum_sin_alpha_sq - sum_sin_alpha**2)
A = (sum_a / N) - B * (sum_sin_alpha / N)
d_i = acceleration - (A + B * sin_alpha)
D = sum_sin_alpha_sq - (sum_sin_alpha**2 / N)
sigma_g = np.sqrt(np.sum(d_i**2) / (D * (N - 2)))
delta_g = 2 * sigma_g
relative_error_g = (delta_g / B) * 100

plt.scatter(sin_alpha[:-4], acceleration[:-4], color='blue', label='Экспериментальные данные')

# 4. Построим график аппроксимирующей прямой: a = A + B sin(alpha)
sin_alpha_theor = np.linspace(min(sin_alpha), max(sin_alpha), 100)
a_theor = A + B * sin_alpha_theor
plt.plot(sin_alpha_theor, a_theor, color='red', label=f'Аппроксимация: a = {A:.2f} + {B:.2f} sin(α)')
plt.errorbar(sin_alpha[:-4], acceleration[:-4], yerr=delta_a[:-4], fmt='o', color='b', label='Погрешности')
# Настройка графика
plt.xlabel('sin(α)')
plt.ylabel('Ускорение a (м/с²)')
plt.title('График зависимости a от sin(α)')
plt.legend()
plt.grid(True)
plt.show()
# Вывод результатов
print(f"Коэффициент B (g): {B}")
print(f"Коэффициент A (-μg): {A:.4f}")
print(f"СКО для g: {sigma_g:.4f}")
print(f"Доверительный интервал для g: {B:.4f} ± {delta_g:.4f}")
print(f"Относительная погрешность g: {relative_error_g:.2f}%")
print(f'Абсолютная погрешность: {delta_g} м/с²')
