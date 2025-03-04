import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Данные
dP = [
    np.array([14,12.8,11.9,11,10.3,9.8,9,8.4,7.8,7.2,6.7,6.2,5.8,5.4,5.1,4.7,4.4,4.1,3.8,3.5,3.3,3]),
    np.array([14.0,12.7,11.8,11.0,10.3,9.6,8.9,8.4,7.8,7.3,6.8,6.2,5.8,5.4,5.1,4.7,4.4,4.1,3.8,3.5,3.3,3.0]),
    np.array([14.0,12.7,11.8,10.9,10.2,9.5,8.8,8.1,7.5,7.0,6.6,6.1,5.7,5.3,5.0,4.6,4.3,4.0,3.7,3.4,3.2,3.0])
]

t = [
    np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,20.7]),
    np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,20.9]),
    np.array([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,20.9])
]
L = 237/1000
D = 0.67/1000
H_c = 239/1000
D_c = 93/1000
delta_L = 1/1000
delta_D = 0.01/1000
delta_H_c = 0.5/1000
delta_D_c = 0.5/1000
P_0 = 1.051*10**5
V_c = H_c*D_c**2*np.pi/4

P_0 = 1.051e5
colors = ['#5986e4', '#7d4dc8', '#b01bb3']

# Расчет X для всех серий
X = [np.log(dp/(dp + 2*P_0)) for dp in dP]

# МНК для всех серий
results = []
for i in range(3):
    n = len(t[i])
    x, y = t[i], X[i]
    
    # Коэффициенты
    C = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / (n*np.sum(x**2) - (np.sum(x))**2)
    X0 = (np.sum(y) - C*np.sum(x)) / n
    
    # Погрешности
    y_fit = X0 + C*x
    residuals = y - y_fit
    sigma = np.sqrt(np.sum(residuals**2)/(n-2))
    sigma_C = sigma / np.sqrt(np.sum((x - np.mean(x))**2))
    
    # Доверительный интервал (95%)
    t_student = stats.t.ppf(0.975, n-2)
    conf_interval = [C - t_student*sigma_C, C + t_student*sigma_C]
    
    results.append((C, sigma_C, conf_interval))
    print(C)

# Графики
plt.figure(figsize=(12, 6))

# Точечные данные
for i in range(3):
    plt.scatter(t[i], X[i], color=colors[i], label=f'Серия {i+1}')

# Аппроксимация для первой серии
x_fit = np.linspace(min(t[0]), max(t[0]), 100)
y_fit = results[0][0] * x_fit + (np.mean(X[0]) - results[0][0]*np.mean(t[0]))
plt.plot(x_fit, y_fit, '--', color=colors[0], label='Аппроксимация серии 1')

# Настройки графика
plt.title('Зависимость X(t) с линейной аппроксимацией', fontsize=14)
plt.xlabel('Время t, с', fontsize=12)
plt.ylabel('X = ln(ΔP/(ΔP + 2P₀))', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Вывод результатов
print("Результаты МНК:")
for i, (C, sigma_C, conf) in enumerate(results):
    print(f'Серия {i+1}:')
    print(f'C = {C:.5f} ± {sigma_C:.5f}')
    print(f'Доверительный интервал: [{conf[0]:.5f}, {conf[1]:.5f}]\n')

weights = [1/(sigma_C**2) for (_, sigma_C, _) in results]

# Взвешенное среднее
C_values = [C for (C, _, _) in results]
C_weighted = np.sum(np.array(C_values) * np.array(weights)) / np.sum(weights)

# Погрешность взвешенного среднего
delta_C_weighted = 1 / np.sqrt(np.sum(weights))

print("\nВзвешенное среднее:")
print(f"C = {C_weighted:.5f} ± {delta_C_weighted:.5f}")
print(f"95% ДИ: [{C_weighted - 2*delta_C_weighted:.5f}, {C_weighted + 2*delta_C_weighted:.5f}]")

delta_V_c = abs(V_c * (delta_C_weighted / C))
conf_interval_Vc = [V_c - 2*delta_V_c, V_c + 2*delta_V_c]
print("\nРезультаты для объема сосуда:")
print(f"V_c = {V_c:.5e} м³")
print(f"ΔV_c = {delta_V_c:.5e} м³")
print(f"95% ДИ: [{conf_interval_Vc[0]:.5e}, {conf_interval_Vc[1]:.5e}] м³")

alpha = C_weighted/(2*P_0)
eta = abs(np.pi*(D/2)**4/(16*alpha*V_c*L))
print("Коэфициет вязкости воздуха:", eta)
plt.show()

