import numpy as np
import matplotlib.pyplot as plt


m_0=24.08/1000
m_1=44.05/1000
m_2=64.02/1000
mu_1_1=np.array([393.7, 692.4, 479.7, 396.2, 311.3])
mu_1_2=np.array([349.8, 550.7, 418.3, 336.1, 281.3])
t_1=np.array([55.85, 90.15, 65.35, 52.25, 44.25])
mu_2_1=np.array([496.0, 385.1, 318.4, 518.7, 449.3])
mu_2_2=np.array([446.2, 359.7, 300.0, 478.2, 417.2])
t_2=np.array([35.12, 27.91, 22.81, 36.44, 32.19])
mu_3_1=np.array([614.5, 652.6, 565.4, 468.9, 580.9])
mu_3_2=np.array([562.5, 592.5, 510.2, 442.5, 540.2])
t_3=np.array([27.54, 31.85, 27.21, 23.08, 28.14])
m_m=1.5
r=12.5/100
l=22.5/100
delta_l=1/10000
g=9.81
delta_m=0.01/1000


w_1=np.array([])
w_2=np.array([])
w_3=np.array([])

def make_w(mu_1, mu_2):
    w=np.array([])
    for i in range(0, 5):
        w=np.append(w, (mu_1[i]+mu_2[i])/30)
    return w

w_1=make_w(mu_1_1, mu_1_2)
w_2=make_w(mu_2_1, mu_2_2)
w_3=make_w(mu_3_1, mu_3_2)
#print(w_1)
def calculate_A_and_errors(omega, T):
    A = np.sum(omega * T) / np.sum(omega ** 2)
    sigma_A = np.sqrt(np.sum((T - A * omega) ** 2) / ((5 - 1) * np.sum(omega ** 2)))
    delta_A = 2 * sigma_A
    epsilon_A = (delta_A / A) * 100
    return A, sigma_A, delta_A, epsilon_A

A1, sigma_A1, delta_A1, epsilon_A1 = calculate_A_and_errors(w_1, t_1)
A2, sigma_A2, delta_A2, epsilon_A2 = calculate_A_and_errors(w_2, t_2)
A3, sigma_A3, delta_A3, epsilon_A3 = calculate_A_and_errors(w_3, t_3)
print(f"A1: {A1:.5f}, σ_A1: {sigma_A1:.5f}, ΔA1: {delta_A1:.5f}, ε_A1: {epsilon_A1:.2f}%")
print(f"A2: {A2:.5f}, σ_A2: {sigma_A2:.5f}, ΔA2: {delta_A2:.5f}, ε_A2: {epsilon_A2:.2f}%")
print(f"A3: {A3:.5f}, σ_A3: {sigma_A3:.5f}, ΔA3: {delta_A3:.5f}, ε_A3: {epsilon_A3:.2f}%")
def plot_graph(omega, T, A, title):
    omega=np.sort(omega)
    T=np.sort(T)
    plt.scatter(omega, T, label="Экспериментальные данные", color="blue")
    plt.plot(omega, A * omega, label=f"Линейная зависимость: A={A:.5f}", color="red")
    plt.xlabel("Угловая скорость, рад/с")
    plt.ylabel("Период прецессии, с")
    plt.title(title)
    plt.legend()
    plt.grid()
    #plt.show()

plot_graph(w_1, t_1, A1, "График для первого момента силы")
plot_graph(w_2, t_2, A2, "График для второго момента силы")
plot_graph(w_3, t_3, A3, "График для третьего момента силы")
print('I_exp=', A1*(m_0)*g*l/(2*np.pi))
print('I_exp=', A2*(m_1)*g*l/(2*np.pi))
print('I_exp=', A3*(m_2)*g*l/(2*np.pi))
print('I_mean_exp=', (A1*(m_0)*g*l/(2*np.pi)+A2*(m_1)*g*l/(2*np.pi)+A3*(m_2)*g*l/(2*np.pi))/3)

I_exp=(A1*(m_0)*g*l/(2*np.pi)+A2*(m_1)*g*l/(2*np.pi)+A3*(m_2)*g*l/(2*np.pi))/3
A=(A1+A2+A3)/3
delta_A=(delta_A1+delta_A2+delta_A3)/3
m=(m_0+m_1+m_2)/3
relative_error = np.sqrt((delta_A / A)**2 + (delta_m / m)**2 + (delta_l / l)**2)
delta_I_exp = relative_error * I_exp
epsilon_I = (delta_I_exp / I_exp) * 100
print('delta_I:', delta_I_exp)
print('I_theory=', m_m*r**2/2)
print(epsilon_I)
