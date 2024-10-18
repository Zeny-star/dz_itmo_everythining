import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing

# Данные
L = 357.8 / 1000  # длина, одна фиксированная величина
delta_L = 0.00007

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
x_values = np.arange(1, 7)

# Применение экспоненциального сглаживания для t1 и t2
smoothing_t1 = ExponentialSmoothing(mean_t1, trend=None, seasonal=None, seasonal_periods=None).fit(smoothing_level=0.3)
smoothing_t2 = ExponentialSmoothing(mean_t2, trend=None, seasonal=None, seasonal_periods=None).fit(smoothing_level=0.3)

# Получение сглаженных значений
smooth_t1 = smoothing_t1.fittedvalues
smooth_t2 = smoothing_t2.fittedvalues

# Построение графиков с учетом погрешностей (полосы)
plt.figure(figsize=(10, 6))

# Создаем полосы шириной 2 * delta_t для t1 и t2
plt.fill_between(x_values, np.array(mean_t1) - 2*delta_t, np.array(mean_t1) + 2*delta_t, color='red', alpha=0.3, label='Полоса t1 (2δt)')
plt.fill_between(x_values, np.array(mean_t2) - 2*delta_t, np.array(mean_t2) + 2*delta_t, color='blue', alpha=0.3, label='Полоса t2 (2δt)')

# Плавные линии на основе экспоненциального сглаживания
plt.plot(x_values, smooth_t1, 'r-', label='Аппроксимация t1 (экспоненциальное сглаживание)')
plt.plot(x_values, smooth_t2, 'b-', label='Аппроксимация t2 (экспоненциальное сглаживание)')

# Подписи и легенда
plt.xlabel('Номер измерения')
plt.ylabel('Среднее время (с)')
plt.title('График времени с учётом погрешностей (экспоненциальное сглаживание)')
plt.legend()
plt.grid()
plt.show()

# Вывод погрешности
print(f'Средняя погрешность времени: {delta_t:.6f} с')
