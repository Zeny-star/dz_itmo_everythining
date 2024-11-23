import matplotlib.pyplot as plt
import numpy as np

# Начальные значения
c_new = 1.0  # Начальное значение для c_x
a_new = 2.0  # Начальное значение для a_x

# Количество точек, которые нужно сгенерировать
num_points = 20

# Списки для хранения значений c_x и a_x
c_values = [1.0]
a_values = [2.0]

# Генерация следующих точек
for i in range(1, num_points):
    # Вычисляем новое значение a_x
    a_new = a_new - (a_new * c_new) / 2
    print(a_new)
    a_values.append(a_new)
    # Вычисляем новое значение c_x
    c_new = (c_new ** 2 * (c_new - 3)) / 4
    c_values.append(c_new)
# Подготовка данных для построения графика
x_values = np.arange(num_points)  # Ось X будет просто индексами точек

# Построение графика
plt.plot(x_values, a_values, marker='o', linestyle='-', color='blue', label='График a_x')
plt.title('График a_x в зависимости от индекса')
plt.xlabel('Индекс (n)')
plt.ylabel('a_x')
plt.axhline(0, color='black', linewidth=0.5, ls='--')
plt.axvline(0, color='black', linewidth=0.5, ls='--')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()

# Выводим все точки на график
# for index, value in enumerate(a_values):
#  plt.text(x_values[index], value, f'({x_values[index]}, {value:.2f})')

# Настройки отображения осей
plt.xlim(0, num_points - 1)
plt.ylim(1, 2)  # Устанавливаем пределы оси Y от -1 до 1
plt.show()
