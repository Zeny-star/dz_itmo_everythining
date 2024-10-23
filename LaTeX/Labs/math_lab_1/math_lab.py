import numpy as np
import matplotlib.pyplot as plt

# Функция для вычисления членов последовательности
def iterate_sequence(a, x0, n):
    x_values = [x0]
    for i in range(1, n):
        x_next = 0.5 * (x_values[-1] + a / x_values[-1])
        x_values.append(x_next)
    return x_values

# Функция для вычисления ошибок
def calculate_errors(x_values, a):
    sqrt_a = np.sqrt(a)
    errors = [abs(x - sqrt_a) for x in x_values]
    return errors

# Параметры
a = 4  # значение a
x0_values = [0.1, 1, 2, 10]  # различные начальные условия
n = 10  # количество итераций

# Построим графики последовательности и ошибок
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Построение графиков для различных значений x0
for x0 in x0_values:
    x_values = iterate_sequence(a, x0, n)
    errors = calculate_errors(x_values, a)
    
    # График последовательности
    axs[0].plot(range(1, n+1), x_values, marker='o', label=f'x0 = {x0}, a = {a}')
    
    # График ошибок
    axs[1].plot(range(1, n+1), errors, marker='o', label=f'x0 = {x0}, a = {a}')

# Настройки графика для последовательности
axs[0].set_title('Графики членов последовательности')
axs[0].set_xlabel('Номер итерации (n)')
axs[0].set_ylabel('Значения членов последовательности (x_n)')
axs[0].legend()
axs[0].grid(True)

# Настройки графика для ошибок
axs[1].set_title('Графики ошибок последовательности')
axs[1].set_xlabel('Номер итерации (n)')
axs[1].set_ylabel('Ошибка |x_n - sqrt(a)|')
axs[1].set_yscale('log')  # Логарифмическая шкала для наглядности ошибок
axs[1].legend()
axs[1].grid(True)

# Отображение графиков
plt.tight_layout()
plt.show()
