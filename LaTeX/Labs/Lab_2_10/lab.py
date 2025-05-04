import numpy as np
import matplotlib.pyplot as plt


m = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])/1000
time = np.array([50.27, 107.03, 164.35, 219.44, 278.39, 335.15, 385, 440.84, 493.44, 548.15, 598.74, 652.87, 703.9, 757.15, 810.72, 859, 897, 917, 1024.06, 1072.29])
c = 4200
R = 82
U = 232
tao = 120
Delta_t = 10
M = 97/100
p = 200
Delta_p = 5
A_1 = []
A_2 = []

N_okr = []
for i in range(len(m)):
    N_okr.append(c*M*(Delta_t/tao))
N_full = []
for i in range(len(time)):
    N_full.append(U**2/R)
for i in range(len(m)):
    A_2.append((N_full[i] - N_okr[i])*time[i])

print(p*time[-1])
print(p)
print(time[-1])
print(A_2)
plt.figure(dpi = 300)
plt.plot(m, A_2)
plt.title("Зависимость A от m")
plt.xlabel("m, кг")
plt.ylabel("A, Дж")
plt.show()

