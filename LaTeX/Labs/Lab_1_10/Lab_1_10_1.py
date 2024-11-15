import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
t_values = np.array([0, 1.622, 3.243, 4.865, 6.487, 8.108, 9.73, 11.352, 12.973, 14.595, 16.217])
A_0_0 = 19.0
A_0 = np.array([19.0, 18.9, 18.4, 18.1, 18.0, 17.8, 17.3, 17.0, 16.9, 16.5, 16.2])
A_200 = np.array([19.0, 16.1, 15.0, 13.8, 12.6, 11.6, 10.4, 9.6, 8.8, 7.9, 7.0])
A_400 = np.array([19.0, 15.0, 11.0, 8.6, 6.4, 4.0, 3.2, 2.2, 1.6, 1.2, 0.8])

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

U_B = np.array([7.5, 8.0, 8.5, 9.0])  # Напряжение
T = np.array([9.25, 8.69, 8.12, 7.69])  # Период колебаний (в секундах)

omega_exp = 2 * np.pi / T  # Угловая частота из периода

theta_0 = 0.3  # Амплитуда угловых колебаний, можно уточнить при необходимости

omega_theory = np.linspace(0, 2 * omega_exp.mean(), 500)  # Для теоретического графика



# График АЧХ
plt.figure(figsize=(12, 6))
for beta in betas:
    a_omega = (omega_exp.mean()**2 * theta_0) / np.sqrt((omega_exp.mean()**2 - omega_theory**2)**2 + 4 * beta**2 * omega_theory**2)
    plt.plot(omega_theory, a_omega, label=f'β = {beta}')
plt.xlabel('ω, рад/с')
plt.ylabel('a(ω)')
plt.title('Амплитудно-частотная характеристика (АЧХ)')
plt.legend()
plt.grid()
plt.show()
print(betas)

