import matplotlib.pyplot as plt
import numpy as np
values = [4.91, 4.81, 4.90, 4.90, 4.84, 5.03, 4.91, 4.90, 5.06, 5.00, 4.93, 4.65, 4.78, 5.06, 4.81, 4.84, 5.03, 4.97, 4.66, 5.07,
          4.81, 4.72, 4.60, 5.12, 4.97, 4.94, 4.91, 4.75, 5.09, 5.00, 5.34, 4.62, 5.03, 5.10, 4.94, 4.78, 4.72, 4.94, 4.84, 4.87,
          5.00, 5.00, 4.93, 4.97, 4.91, 5.03, 4.87, 4.75, 5.00, 4.85]
x = np.linspace(min(values), max(values), 100)
y = 1 / (np.std(values)* np.sqrt(2 * np.pi)) * np.exp(-(x - np.mean(values))**2 / (2 * np.std(values)**2))
confidence_interval = 1.96 * np.std(values) / np.sqrt(len(values))
plt.plot(x, y, color='orange', label='Нормальное распределение')
plt.hist(values, density=True, color='b', label='Гистограмма')
plt.xlabel('Время (секунды)')
plt.ylabel('Плотность')
plt.title('Гистограмма и нормальное распределение')
plt.legend(loc='best')

plt.show()
print('дисперсия',np.std(values))
print('среднее',np.mean(values))
print('разброс', confidence_interval)


