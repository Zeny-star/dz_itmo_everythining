import numpy as np

a = 2
raz = 100
f = np.linspace(0, np.pi/2, raz)
summ =0
for i in f:
    summ += (3*a*(np.cos(i)*np.sin(i))/(np.sin(i)**3 + np.cos(i)**3))**2*(np.pi/(2*raz))

print(summ/2)


