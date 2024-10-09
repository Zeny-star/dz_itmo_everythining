import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Данные для R^2 и I
R_squared = np.array([0.005929, 0.010404, 0.016129, 0.023104, 0.031329, 0.040804])
I_values=np.array([0.0237, 0.0289, 0.0374, 0.0474, 0.0639, 0.0782])


# Линейная регрессия для нахождения коэффициентов и их погрешностей
slope, intercept, r_value, p_value, std_err = stats.linregress(R_squared, I_values)

# Количество точек
n = len(R_squared)

# Найдем необходимые суммы для расчетов дисперсий
mean_R_squared = np.mean(R_squared)
mean_I = np.mean(I_values)

S_xx = np.sum((R_squared - mean_R_squared)**2)  # S_xx
S_yy = np.sum((I_values - mean_I)**2)  # S_yy

# Погрешность свободного члена
intercept_error = np.sqrt(S_yy / ((n - 2) * S_xx))

# Погрешность углового коэффициента (наклона)
slope_error = std_err

# Погрешность четверти углового коэффициента
quarter_slope_error = slope_error / 4

# Вывод результатов
print(f"Угловой коэффициент (slope) = {slope:.4f} ± {slope_error:.4f}")
print(f"Четверть углового коэффициента = {slope / 4:.4f} ± {quarter_slope_error:.4f}")
print(f"Свободный член (intercept) = {intercept:.4f} ± {intercept_error:.4f}")

# Построим график зависимости I(R^2) с нанесенными погрешностями
plt.errorbar(R_squared, I_values, yerr=std_err, fmt='o', label='Экспериментальные данные')
plt.plot(R_squared, intercept + slope * R_squared, label='Линейная регрессия')
plt.xlabel('$R^2$')
plt.ylabel('$I$')
plt.legend()
plt.grid(True)
plt.show()
