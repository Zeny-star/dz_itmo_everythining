import numpy as np


x_1_1 = np.array([1.67, 1.37, 1.54, 1.63, 1.82])
x_1_2 = np.array([6.1, 5.92, 6.09, 6.1, 6.28])
t_1 = 23.62

x_2_1 = np.array([0.33, 0.31, 0.27, 0.31, 0.4])
x_2_2 = np.array([7.55, 7.5, 7.53, 7.56, 7.27])
t_2 = 9.75

x_3_1 = np.array([0.64, 0.56, 0.56, 0.54, 0.7])
x_3_2 = np.array([8.53, 8.62, 8.68, 8.6, 8.53])
t_3 = 6.75

alpha = 0.215/1000
delta_alpha = 0.02/1000
R = 4.40/100
delta_R = 0.05/100
p_liquid = 0.96*1000
delta_p_liquid = 0.01*1000
p_ball = 7.8*1000
delta_p_ball = 0.1*1000
l = 10.3/100
delta_l = 0.1 / 100
g = 9.81
delta_g = 0.01


d_1 = x_1_2 - x_1_1
d_2 = x_2_2 - x_2_1
d_3 = x_3_2 - x_3_1

r_1 = alpha*np.mean(d_1)/2
r_2 = alpha*np.mean(d_2)/2
r_3 = alpha*np.mean(d_3)/2

v_1 = l/t_1
v_2 = l/t_2
v_3 = l/t_3

k_1 = 1/(1+2.4*r_1/R)
k_2 = 1/(1+2.4*r_2/R)
k_3 = 1/(1+2.4*r_3/R)

eta_1 = 2/9*r_1**2*(p_ball-p_liquid)*g/v_1*k_1
eta_2 = 2/9*r_2**2*(p_ball-p_liquid)*g/v_2*k_2
eta_3 = 2/9*r_3**2*(p_ball-p_liquid)*g/v_3*k_3
print(eta_1, eta_2, eta_3)
#delta_v_1 = v_1*np.sqrt((delta_l/l)**2+(delta_t_1/t_1)**2)
#delta_eta_1 = eta_1*np.sqrt((2*delta_R/R)**2+(delta_alpha/alpha)**2+ )

def calc_delta_d(d):
    N = len(d)
    K_s = 2.776  # Для N=5 и доверительной вероятности 95%
    delta_d = K_s * np.sqrt(np.sum((d - np.mean(d))**2) / (N * (N - 1)))
    return delta_d

delta_d1 = calc_delta_d(d_1)
delta_d2 = calc_delta_d(d_2)
delta_d3 = calc_delta_d(d_3)

# Относительные погрешности (формулы 14-15)
delta_r_over_r1 = np.sqrt((delta_alpha / alpha)**2 + (delta_d1 / np.mean(d_1))**2)
delta_r_over_r2 = np.sqrt((delta_alpha / alpha)**2 + (delta_d2 / np.mean(d_2))**2)
delta_r_over_r3 = np.sqrt((delta_alpha / alpha)**2 + (delta_d3 / np.mean(d_3))**2)

delta_v_over_v1 = np.sqrt((delta_l / l)**2 + (0.01 / t_1)**2)  # Погрешность времени 0.01 с
delta_v_over_v2 = np.sqrt((delta_l / l)**2 + (0.01 / t_2)**2)
delta_v_over_v3 = np.sqrt((delta_l / l)**2 + (0.01 / t_3)**2)

delta_rho_term = (delta_p_liquid**2 + delta_p_ball**2) / (p_ball - p_liquid)**2

# Погрешности вязкости (формула 13)
delta_eta1 = eta_1 * np.sqrt(
    (2 * delta_r_over_r1)**2 + 
    delta_v_over_v1**2 + 
    (delta_g / g)**2 + 
    delta_rho_term
)

delta_eta2 = eta_2 * np.sqrt(
    (2 * delta_r_over_r2)**2 + 
    delta_v_over_v2**2 + 
    (delta_g / g)**2 + 
    delta_rho_term
)

delta_eta3 = eta_3 * np.sqrt(
    (2 * delta_r_over_r3)**2 + 
    delta_v_over_v3**2 + 
    (delta_g / g)**2 + 
    delta_rho_term
)
print(delta_eta1, delta_eta2, delta_eta3)
