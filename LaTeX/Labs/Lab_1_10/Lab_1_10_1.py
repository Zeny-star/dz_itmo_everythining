import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import linregress
from scipy.stats import t
from scipy.optimize import fsolve


# Исходные данные
t_values = np.array([0, 1.622, 3.243, 4.865, 6.487, 8.108, 9.73, 11.352, 12.973, 14.595, 16.217])
A_0_0 = 19.0
A_0 = np.array([19.0, 18.9, 18.4, 18.1, 18.0, 17.8, 17.3, 17.0, 16.9, 16.5, 16.2])
A_200 = np.array([19.0, 16.1, 15.0, 13.8, 12.6, 11.6, 10.4, 9.6, 8.8, 7.9, 7.0])
A_400 = np.array([19.0, 15.0, 11.0, 8.6, 6.4, 4.0, 3.2, 2.2, 1.6, 1.2, 0.8])


T = A_0

# 2. Средний период
T_mean = np.mean(T)

# 3. Погрешность среднего периода
n = len(T)  # количество измерений
T_std = np.std(T, ddof=1)  # стандартное отклонение
t_crit = t.ppf(0.95, df=n-1)  # критическое значение t-распределения для доверительной вероятности 0.9
T_error = t_crit * T_std / np.sqrt(n)  # погрешность среднего периода

# 4. Циклическая частота
omega = 20 * np.pi / T_mean

# 5. Погрешность циклической частоты
omega_error = omega * T_error / T_mean  # по формуле погрешностей

# Результаты
print(f"Средний период T: {T_mean:.4f} ± {T_error:.4f} с")
print(f"Циклическая частота ω: {omega:.4f} ± {omega_error:.4f} рад/с")
# Рассчет ln(A0_0/A) для каждой серии
lnA_0_0_A_0 = np.log(A_0_0 / A_0)
lnA_0_0_A_200 = np.log(A_0_0 / A_200)
lnA_0_0_A_400 = np.log(A_0_0 / A_400)

# Вычисление лямбда для каждого интервала (λ = ln(An/An+1))
lambda_0 = np.log(A_0[:-1] / A_0[1:])
lambda_200 = np.log(A_200[:-1] / A_200[1:])
lambda_400 = np.log(A_400[:-1] / A_400[1:])

# Средние значения лямбда
lambda_0_avg = np.mean(lambda_0)
lambda_200_avg = np.mean(lambda_200)
lambda_400_avg = np.mean(lambda_400)

print(f'Среднее значение λ для A_0: {lambda_0_avg:.5f}')
print(f'Среднее значение λ для A_200: {lambda_200_avg:.5f}')
print(f'Среднее значение λ для A_400: {lambda_400_avg:.5f}')
print(f'Среднее значение Q для A_0: {np.pi/lambda_0_avg:.5f}')
print(f'Среднее значение Q для A_200: {np.pi/lambda_200_avg:.5f}')
print(f'Среднее значение Q для A_400: {np.pi/lambda_400_avg:.5f}')

# Создание графика функции f(t) = ln(A0_0 / A_k)
plt.figure(figsize=(10, 6))
plt.scatter(t_values, lnA_0_0_A_0, label='ln(A_0_0 / A_0)')
plt.scatter(t_values, lnA_0_0_A_200, label='ln(A_0_0 / A_200)')
plt.scatter(t_values, lnA_0_0_A_400, label='ln(A_0_0 / A_400)')
betas=[]
# Линейная аппроксимация графиков ln(A_0_0 / A_k) и расчет коэффициентов
for ln_values, label in zip([lnA_0_0_A_0, lnA_0_0_A_200, lnA_0_0_A_400], 
                            ['f(t) для A_0', 'f(t) для A_200', 'f(t) для A_400']):
    A = np.vstack([t_values, np.ones(len(t_values))]).T
    beta, free_coeff = np.linalg.lstsq(A, ln_values, rcond=None)[0]
    betas.append(beta)
    plt.plot(t_values, beta * t_values + free_coeff, label=f'Аппроксимация {label}')
    print(f'Коэффициент затухания (β) для {label}: {beta:.5f}')
    print(f'Свободный коэффициент для {label}: {free_coeff:.5f}')

plt.xlabel('Время (t)')
plt.ylabel('ln(A_0_0 / A_k)')
plt.title('Графики функции f(t) = ln(A_0_0 / A_k)')
plt.legend()
plt.grid()
plt.show()

# График зависимости A(t) от t
plt.figure(figsize=(10, 6))
plt.plot(t_values, A_0, 'o-', label='A(t) для A_0')
plt.plot(t_values, A_200, 'o-', label='A(t) для A_200')
plt.plot(t_values, A_400, 'o-', label='A(t) для A_400')



plt.xlabel('Время (t)')
plt.ylabel('Амплитуда (A)')
plt.title('Зависимость амплитуды A(t) от времени')
plt.legend()
plt.grid()
plt.show()

U = np.array([7.5, 8.0, 8.5, 9.0])  # Напряжение
T = np.array([9.25, 8.69, 8.12, 7.69])  # Период колебаний (в секундах) - это кста для 5 оборотов

omega = 10*np.pi/T
slope, intercept, _, _, _ = linregress(U, omega)

# Генерация точек для аппроксимирующей прямой
U_fit = np.linspace(min(U), max(U), 200)  # Шаги для линии
omega_fit = slope * U_fit + intercept    # Значения ω по прямой

# Построение графика
plt.figure(figsize=(8, 5))
plt.plot(U, omega, marker='o', linestyle='', color='b', label='Исходные данные')
plt.plot(U_fit, omega_fit, linestyle='--', color='r', label='Линейная аппроксимация')

# Оформление графика
plt.title(r'Градуировочный график $\omega(U)$')
plt.xlabel('Напряжение $U$ (В)')
plt.ylabel('Угловая частота $\omega$ (рад/с)')
plt.grid(True)
plt.xticks(np.arange(7.5, 9.1, 0.1))
plt.yticks(np.arange(3.3, 4.1, 0.1))
plt.legend()
plt.show()
interpolation_function = interp1d(U, omega, kind='linear', fill_value="extrapolate")
# Указанные напряжения
U_values = np.array([7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9])

# Получение омеги для всех напряжений
omega_values = interpolation_function(U_values)

# Вывод результата
for voltage, omega_value in zip(U_values, omega_values):
    print(f"Напряжение: {voltage:.1f} В, ω: {omega_value:.4f} рад/с")




x = np.array([
    0.8867442521,
    0.9003690037,
    0.9139937553,
    0.927618507,
    0.9412432586,
    0.9548680102,
    0.9682089129,
    0.9818336645,
    0.9954584161,
    1.009083168,
    1.022707919,
    1.035764973,
    1.049105876,
    1.062446778,
    1.075503832,
    1.088844735,
    1.101050241,
    1.113539597,
    1.125745104,
    1.138234459,
    1.150723815
])

# Исходные данные y
y1 = np.array(
[2.6, 2.7, 2.8, 3.0, 3.8, 5.6, 5.9, 8.4, 17.4, None, None, 13.4, 7.2, 4.6, 4.2, 3.0, 2.8, 2.6, 2.4, 2.2, 2.1]
)
y2 = np.array(
[2.5, 2.4,2.6, 3.0, 3.6, 4.2, 4.8, 6.0, 11.0, 11.8, 12.0, 6.2, 5.4, 4.2, 3.2, 3.0, 2.4, 2.2, 2.0, 1.8, 1.5]
)
y3 = np.array(
[0.5, 1, 1.8, 2.0, 2.2, 2.6, 2.9, 3.2, 3.5, 3.6, 3.4, 3.0, 2.8, 2.4, 2.2, 1.8, 1.7, 1.6, 1.4, 1, 0.8]
)

# Замените None на NaN для корректного отображения
y1 = np.where(y1 == None, np.nan, y1)
y2 = np.where(y2 == None, np.nan, y2)
y3 = y3  # Делим на 0.4

# Создание фигуры и оси
fig, ax = plt.subplots()

# Настройка мелкой разметки
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

# Построение графиков, соединяющих точки
line1, = ax.plot(omega_values, y1, marker='o', color='blue', label='Экспериментальные данные I=100 мА')
line2, = ax.plot(omega_values, y2, marker='o', color='green', label='Экспериментальные данные I=200 мА')
line3, = ax.plot(omega_values, y3, marker='o', color='purple', label='Экспериментальные данные I=400 мА')

# Настройка графика
ax.set_xlabel('$\omega$/$\omega_0$ ')
ax.set_ylabel('$a_{ст}$')
ax.legend(fontsize=8)  # Уменьшение размера шрифта легенды
ax.grid(which='both')  # Отображаем основную и мелкую сетку

# Отображение графика
plt.show()


# Параметры для a(ω)
theta_0 = 0.3
beta = 0.044
omega_0 = 3.523

# Создание массива значений ω
omega = np.linspace(3, 4.5, 300)  # Избегаем 0, чтобы избежать деления на 0

# Расчёт a(ω)
a_omega = (omega_0**2 * theta_0) / np.sqrt((omega_0**2 - omega**2)**2 + (4 * beta**2 * omega**2))

# Преобразование ω в ω/ω_0
omega_scaled = omega / omega_0

# Данные
fig, ax = plt.subplots()

# Настройка мелкой разметки
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

# Построение графиков, соединяющих точки
line2, = ax.plot(x, y2, marker='o', color='green', label='Экспериментальные данные I=200 мА')

# Добавление графика a(ω/ω_0)
ax.plot(omega_scaled, a_omega, color='c', label=r'$a\left(\frac{\omega}{\omega_0}\right)$')

# Настройка графика
ax.set_xlabel(r'$\frac{\omega}{\omega_0}$')
ax.set_ylabel('$a_{ст}$')
ax.legend(fontsize=8)  # Уменьшение размера шрифта легенды
ax.grid(which='both')  # Отображаем основную и мелкую сетку

# Отображение графика
plt.show()

print('Q по АЧХ:', 12/0.4)

