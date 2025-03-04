import numpy as np

m = 4.59/1000
delta_m = 0.02/1000
r = 5.95/1000
delta_r = 0.02/1000
V = 1.14/1000
delta_V = 0.01/1000
p_0 = 1.051*10**5

N = np.array([])
t = np.array([])


T = t/N
sigma = np.sqrt(())

gamma = 4*m*V/(T**2*p_0*r**4)

i = 2/(gamma-1)
print(f'Таблица периодов: {T}')
print(f'Показаетль адиабаты: {gamma}')
print(f'Число степеней свободы: {i}')
