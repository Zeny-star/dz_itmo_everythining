import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

m0_1=48.03
m0_2=99.14
x1 = 0.150
x2 = 0.800
g = 9.82
m=np.array([1.86, 2.75, 3.52, 4.43, 5.28, 6.14, 6.95])
m=m/1000
v1_1=np.array([0.28, 0.34, 0.4, 0.44, 0.48, 0.52, 0.55])
v1_2=np.array([0.57, 0.74, 0.91, 1.01, 1.1, 1.18, 1.26])
v2_1=np.array([0.19, 0.25, 0.29, 0.32, 0.35, 0.35, 0.41])
v2_2=np.array([0.46, 0.5, 0.66, 0.74, 0.81, 0.88, 0.92])
def acceleration_and_tension(v1, v2, x1, x2, m, g=9.82):
    a = (v2**2 - v1**2) / (2 * (x2 - x1))
    T = m * (g - a)
    return a, T

# Вычисления
a_values_1, T_values_1 = acceleration_and_tension(v1_1, v1_2, x1, x2, m)
a_values_2, T_values_2 = acceleration_and_tension(v2_1, v2_2, x1, x2, m)
# График зависимости T от a
plt.scatter(a_values_1, T_values_1, color='blue', label='Экспериментальные данные')
plt.scatter(a_values_2, T_values_2, color='blue', label='Экспериментальные данные')
plt.xlabel('Ускорение, м/с²')
plt.ylabel('Сила натяжения, Н')
plt.title('График зависимости силы натяжения от ускорения')
plt.legend()
plt.grid(True)
plt.show()

# Линейная аппроксимация
def linear_fit(a, M1, F_tr):
    return M1 * a + F_tr

params_1, covariance_1 = curve_fit(linear_fit, a_values_1, T_values_1)
M1_1 = params_1[0]
F_tr_1 = params_1[1]
uncertainties_1 = np.sqrt(np.diag(covariance_1))

params_2, covariance_2 = curve_fit(linear_fit, a_values_2, T_values_2)
M1_2 = params_2[0]
F_tr_2 = params_2[1]
uncertainties_2 = np.sqrt(np.diag(covariance_2))


print(f"Масса тележки M1_1: {M1_1:.5f} кг ± {uncertainties_1[0]:.5f}")
print(f"Сила трения F_tr_1: {F_tr_1:.5f} Н ± {uncertainties_1[1]:.5f}")

print(f"Масса тележки M1_2: {M1_2:.5f} кг ± {uncertainties_2[0]:.5f}")
print(f"Сила трения F_tr_2: {F_tr_2:.5f} Н ± {uncertainties_2[1]:.5f}")


# График с аппроксимацией
a_fit_1 = np.linspace(min(a_values_1), max(a_values_1), 100)
T_fit_1 = linear_fit(a_fit_1, M1_1, F_tr_1)
a_fit_2 = np.linspace(min(a_values_2), max(a_values_2), 100)
T_fit_2 = linear_fit(a_fit_2, M1_2, F_tr_2)


plt.scatter(a_values_1, T_values_1, color='blue', label='Экспериментальные данные')
plt.scatter(a_values_2, T_values_2, color='green', label='Экспериментальные данные')
plt.plot(a_fit_1, T_fit_1, color='red', label=f'Аппроксимация: M1={0.054} кг, F_tr={0.002} Н')
plt.plot(a_fit_2, T_fit_2, color='pink', label=f'Аппроксимация: M1={0.105} кг, F_tr={0.003} Н')
plt.xlabel('Ускорение, м/с²')
plt.ylabel('Сила натяжения, Н')
plt.title('График зависимости силы натяжения от ускорения с аппроксимацией для тележки:(1) - без груза, (2) - с грузом ')
plt.legend()
plt.grid(True)
plt.show()
