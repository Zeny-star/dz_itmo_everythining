import numpy as np

# Заданные данные и погрешности
delta_x = 0.0005
delta_v = 0.01
delta_m = 0.00001
t_alpha_dov = 2.776  # Для N=5 и доверительного уровня 95%

# Данные для абсолютно неупругого соударения
v_1_1 = np.array([0.49, 0.5, 0.49, 0.5, 0.49])
v_1_2 = np.array([0.19, 0.21, 0.21, 0.2, 0.19])
v_2_1 = np.array([0.5, 0.5, 0.5, 0.5, 0.49])
v_2_2 = np.array([0.11, 0.12, 0.1, 0.11, 0.09])

m1_1 = 52.6
m2_1 = 100.03
#100.03
# Вычисление импульсов до и после соударения
def calc(m1, m2, v1, v2):
    p10 = m1 * v1
    p = (m1 + m2) * v2
    delta_p = (p / p10) - 1
    w10 = m1 * v1**2 / 2
    w = (m1 + m2) * v2**2 / 2
    delta_w = (w / w10) - 1
    delta_p_mean = np.mean(delta_p)
    delta_w_mean = np.mean(delta_w)

    delta_p_error = t_alpha_dov * np.sqrt(np.sum((delta_p - delta_p_mean)**2) / (len(delta_p)*(len(delta_p) - 1)))
    delta_w_error = t_alpha_dov * np.sqrt(np.sum((delta_w - delta_w_mean)**2) / (len(delta_w)*(len(delta_w) - 1)))
    th_delta_w=-m2/(m2+m1)
    return delta_p_mean, delta_p_error, delta_w_mean, delta_w_error, th_delta_w



value_delta_p_mean_1, value_delta_p_error_1, value_delta_w_mean_1, value_delta_w_error_1, th_delta_w1 = calc(m1_1, m1_1, v_1_1, v_1_2)
value_delta_p_mean_2, value_delta_p_error_2, value_delta_w_mean_2, value_delta_w_error_2, th_delta_w2= calc(m1_1, m2_1, v_2_1, v_2_2)


print(f"Среднее относительное изменение импульса: {value_delta_p_mean_1:.5f} ± {value_delta_p_error_1:.5f}")
print(f"Среднее относительное изменение энергии: {value_delta_w_mean_1:.5f} ± {value_delta_w_error_1:.5f}")
print(f"Th отклонение энергии:", th_delta_w1)

print(f"Среднее относительное изменение импульса: {value_delta_p_mean_2:.5f} ± {value_delta_p_error_2:.5f}")
print(f"Среднее относительное изменение энергии: {value_delta_w_mean_2:.5f} ± {value_delta_w_error_2:.5f}")
print(f"Th отклонение энергии:", th_delta_w2)

