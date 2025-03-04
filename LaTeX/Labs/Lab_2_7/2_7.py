import numpy as np

m = 4.59/1000
delta_m = 0.02/1000
r = 5.95/1000
delta_r = 0.02/1000
V = 1.14/1000
delta_V = 0.01/1000
p_0 = 1.031*10**5

N = np.array([51, 51, 51, 51, 51])
t = np.array([17.19, 17.75, 17.43, 17.59, 17.15])

T = t/N
n = 5
T_mean = np.mean(T)

mu = np.std(T)/np.sqrt(n)
s = 2.776
delta_T = mu*s


print(r'Таблица периодов:', T, r'$\pm$', delta_T/2)


gamma = 4*m*V/(T**2*p_0*r**4)
gamma_mean = np.mean(gamma)

rel_err_m = delta_m / m
rel_err_V = delta_V / V
rel_err_T = (delta_T/2) / T_mean
rel_err_r = delta_r / r

rel_err_gamma = np.sqrt(rel_err_m**2 + rel_err_V**2 + (2 * rel_err_T)**2 + (4 * rel_err_r)**2)
delta_gamma = gamma_mean * rel_err_gamma
gamma_CI = (gamma_mean - delta_gamma, gamma_mean + delta_gamma)

print("\nПоказатель адиабаты γ:")
print("Вычисленное значение γ: {:.5f}".format(gamma_mean))
print("95%-й доверительный интервал для γ: ({:.5f}, {:.5f})".format(gamma_CI[0], gamma_CI[1]))


i = 2/(gamma-1)
i_mean = np.mean(i)

delta_i = 2 / (gamma_mean - 1)**2 * delta_gamma
i_CI = (i_mean - delta_i, i_mean + delta_i)

print("\nЧисло степеней свободы i:")
print("Вычисленное значение i: {:.5f}".format(i_mean))
print("95%-й доверительный интервал для i: ({:.5f}, {:.5f})".format(i_CI[0], i_CI[1]))

print('Показаетль адиабаты:', gamma)
print('Число степеней свободы:', i)
