import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 19})

times = np.array([78, 171, 207, 245, 289, 331, 376, 421, 470, 521, 575, 631, 691, 753, 821, 892, 969, 1053, 1139, 1239, 1347, 1468, 1605, 1745, 1911, 2102, 2340])
times_c = np.array([75, 172, 219, 267, 318, 372, 427, 485, 543, 608, 678, 744, 820, 894, 977, 1066, 1160, 1264, 1377, 1501, 1643, 1791, 1964, 2168, 2389, 2659])

I = 0.5
U = 7.1
P = I * U
m = 42.5/1000

delta_U = 0.1
delta_I = 0.025
delta_P = np.sqrt((delta_I * U)**2 + (delta_U * I)**2)

times = times - 10

y = []
for i in range(1, len(times)-1):
    y.append(2 / (times[i+1] - times[i-1]))

y_c = []
for i in range(1, len(times_c)-1):
    y_c.append(2 / (times_c[i+1] - times_c[i-1]))

y = np.log(y)
times = times[5:-1]
y = y[4:]

y_c = np.log(y_c)
times_c = times_c[5:-1]
y_c = y_c[4:]

def linear_model(x, a, b):
    return a * x + b

params, _ = curve_fit(linear_model, times, y)

a, b = params
perr = np.sqrt(np.diag(_))

C_0 = P / np.exp(b)

delta_b = perr[1]
delta_C_0 = np.sqrt((delta_P / P)**2 + delta_b**2)

params_c, _c = curve_fit(linear_model, times_c, y_c)

a_c, b_c = params_c
perr_c = np.sqrt(np.diag(_c))

C_0_c = P / np.exp(b_c)

delta_b_c = perr_c[1]
delta_C_0_c = np.sqrt((delta_P / P)**2 + delta_b_c**2)
delta_c = np.sqrt(delta_C_0**2 + delta_C_0_c**2)



plt.plot(times, np.array(y), 'o', label='Экспериментальные данные', color='#5986e4')
plt.plot(times, linear_model(times, a, b), 'r-', label='Аппроксимирующая прямая', color='#b01bb3')
plt.ylabel(r'$\ln\left(\frac{d(T-T_{\text{окр}})}{dt}\right) \, \left[\frac{\degree C}{\text{с}}\right]$', fontsize=20)
#plt.ylabel(r'$\frac{d(T-T_{\text{окр}})}{dt} \, \left[\frac{\degree C}{\text{с}}\right]$', fontsize=20)
plt.xlabel(r'$t \, [\text{c}]$', fontsize=20)
plt.legend(loc='upper right', fontsize=12)
#plt.show()


plt.plot(times_c, np.array(y_c), 'o', label='Экспериментальные данные', color='#5986e4')
plt.plot(times_c, linear_model(times_c, a_c, b_c), 'r-', label='Аппроксимирующая прямая', color='#b01bb3')
plt.ylabel(r'$\ln\left(\frac{d(T-T_{\text{окр}})}{dt}\right) \, \left[\frac{\degree C}{\text{с}}\right]$', fontsize=20)
#plt.ylabel(r'$\frac{d(T-T_{\text{окр}})}{dt} \, \left[\frac{\degree C}{\text{с}}\right]$', fontsize=20)
plt.xlabel(r'$t \, [\text{c}]$', fontsize=20)
plt.legend(loc='upper right', fontsize=12)
#plt.show()


print(f'Коэффициенты линейной зависимости: a = {a}, b = {b}')
print(f'C_0 = {C_0}')
print(f'погрешсть C_0 = {delta_C_0}')

print(f'Коэффициенты линейной зависимости: a = {a_c}, b = {b_c}')
print(f'C_0 = {C_0_c}')
print(f'Погрешсть C_0 = {delta_C_0_c}')
print(f'Удельная теплоемкость образца: {(C_0_c-C_0)/m}')
print(f'Погрешность удельной теплоемкости: {delta_c}')

print(f'Частное угловых коэффициентов: {a/a_c}, часное (C_0+C)/C={(C_0_c)/C_0}')
