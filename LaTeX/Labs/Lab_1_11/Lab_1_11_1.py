import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import math
from statsmodels.tsa.api import ExponentialSmoothing
# Данные
L = 361.8 / 1000  # длина, одна фиксированная величина
delta_L = 0.00007
N=10

t1 = [
    10.391, 10.414, 10.429,
    10.981, 10.987, 11.007,
    11.698, 11.696, 11.701,
    12.442, 12.425, 12.456,
    13.376, 13.378, 13.409,
    14.522, 14.566, 14.605,
]

t2 = [
    11.787, 11.787, 11.794,
    11.900, 11.900, 11.904,
    11.995, 11.994, 11.997,
    12.094, 12.108, 12.111,
    12.215, 12.220, 12.215,
    12.317, 12.323, 12.232
]

# Вычисление средних значений времени
mean_t1 = [np.mean(t1[3*i:3*i+3]) for i in range(6)][::-1]
mean_t2 = [np.mean(t2[3*i:3*i+3]) for i in range(6)][::-1]

# Стандартные отклонения для расчета погрешности delta_t
delta_t1 = [np.std(t1[3*i:3*i+3], ddof=1) for i in range(6)][::-1]
delta_t2 = [np.std(t2[3*i:3*i+3], ddof=1) for i in range(6)][::-1]

# Средняя погрешность времени
delta_t = (np.mean(delta_t1) + np.mean(delta_t2)) / 2

# Значения оси X (для различного t)
x_values = np.arange(2, 8)

# Плавная аппроксимация (полином 3-й степени)
x_smooth = np.linspace(x_values.min(), x_values.max(), 50000)
poly_t1 = np.poly1d(np.polyfit(x_values, mean_t1, 3))
poly_t2 = np.poly1d(np.polyfit(x_values, mean_t2, 3))

coeffs_t1 = np.polyfit(x_values, mean_t1, 3)
coeffs_t2 = np.polyfit(x_values, mean_t2, 3)

def intersection(x):
    return np.polyval(coeffs_t1, x) - np.polyval(coeffs_t2, x)

# Нахождение точки пересечения
x_intersect = fsolve(intersection, 5)[0]
y_intersect = np.polyval(coeffs_t1, x_intersect)
delta_t0 = delta_t
delta_g = np.sqrt(delta_L**2 + (2 * delta_t0)**2)

# Построение графиков с учетом погрешностей (полосы)
plt.figure(figsize=(10, 6))

# Создаем полосы шириной 2 * delta_t для t1 и t2
plt.fill_between(x_values, np.array(mean_t1) - 2*delta_t, np.array(mean_t1) + 2*delta_t, color='red', alpha=0.3, label='Полоса t1 (2δt)')
plt.fill_between(x_values, np.array(mean_t2) - 2*delta_t, np.array(mean_t2) + 2*delta_t, color='blue', alpha=0.3, label='Полоса t2 (2δt)')
plt.scatter(x_intersect, y_intersect, c='g', marker='o', label='Точка пересечения')
# Плавные линии на основе полиномов
plt.plot(x_smooth, poly_t1(x_smooth), 'r-', label='Аппроксимация t1')
plt.plot(x_smooth, poly_t2(x_smooth), 'b-', label='Аппроксимация t2')
plt.plot(x_values, mean_t1, c='r', marker='x', label='Среднее t1')
plt.plot(x_values, mean_t2, c='b', marker='x', label='Среднее t2')
plt.axhline(y=y_intersect, color='gray', linestyle='--')
plt.axvline(x=x_intersect, color='gray', linestyle='--')

# Подписи и легенда
plt.xlabel('Отступ от начала стержня (см)')
plt.ylabel('Время 10 периодов(с)')
plt.title('График времени с учётом погрешностей (плавный)')
plt.legend()
plt.grid()
plt.show()
smoothing_t1 = ExponentialSmoothing(mean_t1, trend=None, seasonal=None, seasonal_periods=None).fit(smoothing_level=0.3)
smoothing_t2 = ExponentialSmoothing(mean_t2, trend=None, seasonal=None, seasonal_periods=None).fit(smoothing_level=0.3)

# Получение сглаженных значений
smooth_t1 = smoothing_t1.fittedvalues
smooth_t2 = smoothing_t2.fittedvalues
t0 = (np.mean(smooth_t1) + np.mean(smooth_t2)) / 2
g=4*math.pi**2*L*(N/y_intersect)**2
relative_error_g = np.sqrt(delta_L**2 + (2 * delta_t / t0)**2)
absolute_error_g = g * relative_error_g
# Вывод погрешности
print(f'Средняя погрешность времени: {delta_t:.6f} с')
print(f'Координата L в точке пересечения: {x_intersect:.3f} см')
print(f'Координата y в точке пересечения: {y_intersect:.3f} с')
print(f'Относительная погрешность delta_g: {delta_g*100:.6f}%')
print(f'Относительная погрешность g: {relative_error_g*100:.6f}%')
print(f'Абсолютная погрешность g: {absolute_error_g:.6f} м/с²')
print(f'g=',g)
