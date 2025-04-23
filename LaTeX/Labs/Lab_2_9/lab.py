import numpy as np
import matplotlib.pyplot as plt

U_n = np.array([1.0, 1.7, 2.4, 3.1, 3.8, 4.5, 5.2, 5.9, 6.6, 7.0])
U_0 = np.array([27.6, 44.9, 60.9, 72.9, 87.3, 100.1, 111.2, 120.3, 128.5, 132.8])
I_n = np.array([0.184, 0.299333333, 0.406, 0.486, 0.582, 0.667333333, 0.741333333, 0.802, 0.856666667, 0.885333333])
R_0 = 0.15
L = 420/1000
delta_L = 0.1/1000
D = 0.1
r_1 = D/2
D_in = 0.6/1000
r_2 = D_in/2
delta_T = 0.5
delta_x = 15/100
alpha = 3.9/1000

P = U_n * I_n
R_n = U_n / I_n

color1 = '#5986e4' # Синеватый
color2 = '#7d4dc8' # Фиолетовый
color3 = '#b01bb3' # Пурпурный/Розовый


selected_R = R_n[:5]
selected_P = P[:5]

gamma, intercept = np.polyfit(selected_R, selected_P, 1)
cov = np.cov(selected_P, selected_R, ddof=0)
delta_gamma = np.sqrt(cov[0,0]/cov[1,1])/len(selected_R)

R_star = -intercept/gamma
R_fit = np.linspace(R_star, R_n.max(), 100)
P_fit = gamma * R_fit + intercept

plt.figure(figsize=(10,5))
plt.plot(R_n, P, 'o', color=color1)
plt.plot(R_fit, P_fit, '--', color=color2)
plt.axvline(R_star, color='green', linestyle='--')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('R, Ом')
plt.ylabel('P, Вт')
plt.show()

X_cp = R_star * alpha * np.log(r_1/r_2)/(2*np.pi*L*gamma)*1000

print(f"⟨γ⟩ = {gamma:.2f} ± {delta_gamma:.2f}")
print(f"R* = {R_star:.5f}")
print(f'X_cp = {X_cp:.5f}')
