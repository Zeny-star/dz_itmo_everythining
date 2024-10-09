import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Пример экспериментальных данных
x1 = 0.15
x2 = [0.50, 0.70, 0.90, 1.10, 1.20]
t1 = [1.2, 1.2, 1.2, 1.2, 1.3]
t2 = [2.8, 3.3, 3.9, 4.4, 4.6]
Y=[]
Z = []
for i in x2:
    Y.append(i-x1)
for j in range(5):
    Z.append((t2[j]**2-t1[j]**2)/2)
Y=np.array(Y)
Z=np.array(Z)

error_Y = np.array([0.05, 0.05, 0.05, 0.05, 0.05])  # Погрешности для Y
error_Z = np.array([0.30,0.35,0.41,0.46,0.48])  # Погрешности для Z

# Линейная регрессия для нахождения коэффициента a
coeffs = np.polyfit(Z, Y, 1)  # Подгонка линейной зависимости
a = coeffs[0]  # Угловой коэффициент (ускорение)
b = coeffs[1]  # Свободный член

# Теоретическая прямая
Z_theor = np.linspace(min(Z), max(Z), 100)
Y_theor = a * Z_theor + b

# Построение графика
plt.figure(figsize=(8, 6))
plt.errorbar(Z, Y, yerr=error_Y, xerr=error_Z, fmt='o', label='Экспериментальные данные', capsize=5, elinewidth=1, markeredgewidth=1)

# Построение теоретической прямой
plt.plot(Z_theor, Y_theor, label=f'Теоретическая зависимость: Y = {a:.2f}Z + {b:.2f}', color='red')
alpha = 0.9
a = np.sum(Z * Y) / np.sum(Z**2)
sigma_a = np.sqrt(1 / (len(Z) - 1) * np.sum((Y - a * Z)**2) / np.sum(Z**2))
# Квантиль распределения Стьюдента для уровня значимости 0.90 и (n-1) степеней свободы
t_alpha = stats.t.ppf((1 + alpha) / 2, df=len(Z)-1)
# Рассчёт относительной погрешности
delta_a = t_alpha * sigma_a
relative_error_a = (delta_a / a) * 100
print(a, 'коэффициент')
print(sigma_a, 'среднее квадратичное отклонение')
print(delta_a, 'абсолютная погрешность')
print(relative_error_a,'%', 'относительная погрешность')
print(f"Доверительный интервал для ускорения: [{a-delta_a}, {a+delta_a}]")
# Оформление графика

plt.xlabel('Z')
plt.ylabel('Y')
plt.title('Зависимость Y от Z с погрешностями')
plt.legend()
plt.grid(True)

# Показать график
plt.show()
