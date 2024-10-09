import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Данные для R^2 и I
R_squared = np.array([0.005929, 0.010404, 0.016129, 0.023104, 0.031329, 0.040804])
I_values=np.array([0.0237, 0.0289, 0.0374, 0.0474, 0.0639, 0.0782])


# Погрешности измерений (заданы вручную или вычислены)
I_errors = np.array([0.002, 0.002, 0.002, 0.002, 0.002, 0.002])  # погрешности I
R_squared_errors = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])  # погрешности R^2

# Функция для аппроксимации (линейная зависимость)
def linear_func(x, a, b):
    return a * x + b

# Аппроксимация данных (линейная регрессия)
popt, pcov = curve_fit(linear_func, R_squared, I_values, sigma=I_errors, absolute_sigma=True)

# popt содержит коэффициенты a (наклон) и b (свободный член)
a, b = popt
print(a/4)
# pcov - ковариационная матрица
perr = np.sqrt(np.diag(pcov))  # Стандартные ошибки коэффициентов

# Стандартная ошибка углового коэффициента и свободного члена
error_a = perr[0]  # Погрешность наклона
error_b = perr[1]  # Погрешность свободного члена

# Рассчитаем четверть от погрешности углового коэффициента
error_a_quarter = error_a / 4

# Вывод результатов
print(f"Угловой коэффициент (I): {a:.4f} ± {error_a:.4f}")
print(f"Свободный член (Mтр): {b:.4f} ± {error_b:.4f}")
print(f"Четверть погрешности углового коэффициента: {error_a_quarter:.4f}")

# Построение графика зависимости I(R^2)
plt.errorbar(R_squared, I_values, yerr=I_errors, xerr=R_squared_errors, fmt='o', label='Экспериментальные данные с погрешностями')

# Линия аппроксимации
plt.plot(R_squared, linear_func(R_squared, *popt), label=f'Аппроксимация: I = {a:.4f} R^2 + {b:.4f}')

# Оформление графика
plt.xlabel(r'$R^2$')
plt.ylabel(r'$I$')
plt.title('Зависимость I(R^2) с погрешностями')
plt.legend()
plt.grid(True)

# Показать график
plt.show()
