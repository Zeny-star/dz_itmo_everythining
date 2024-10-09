import matplotlib.pyplot as plt
import numpy as np

# Исходные данные
h = 0.7  # расстояние, м
delta_h = 0.0005  # погрешность расстояния, м
delta_t = 0.1  # Погрешность времени (можно взять оценочно)
d = 0.046  # Диаметр ступицы в метрах
delta_d = 0.0005
m_karetki = 0.047
delta_m_karetki = 0.0005
m_shaibi=0.22
delta_m=0.0005
delta_m_shaibi=0.0005
# Время для каждого груза (m1, m2, m3, m4)
t1 = [4.47, 4.47, 4.53,
      3.25, 3.31, 3.32,
      2.56, 2.50, 2.63,
      2.32, 2.24, 2.32]
t2 = [5.38, 5.35, 5.50,
      3.94, 3.94, 3.81,
      3.15, 3.13, 3.07,
      2.75, 2.72, 2.84]
t3 = [6.28, 6.37, 6.32,
      4.53, 4.57, 4.53,
      3.65, 3.54, 3.66,
      3.29, 3.32, 3.36]
t4 = [7.41, 7.49, 7.40,
      5.41, 5.34, 5.35,
      4.44, 4.37, 4.38,
      3.81, 3.81, 3.83]
t5 = [8.55, 8.50, 5.58,
      6.09, 6.12, 6.09,
      5.06, 5.07, 5.13, 
      4.19, 4.25, 4.24]
t6 = [10.03, 9.92, 9.94,
      6.88, 6.84, 6.81,
      5.60, 5.56, 5.54,
      4.75, 4.72, 4.76]
m_shaibi = [m_shaibi, m_shaibi*2, m_shaibi*3, m_shaibi*4]

# Объединение данных
times = [t1, t2, t3, t4, t5, t6]

# Рассчет среднего времени
t_mean = [np.mean(t[i:i+3]) for t in times for i in range(0, len(t), 3)]

# Функция для вычисления ускорения
def calculate_acceleration(h, t_mean):
    return [2 * h / (t ** 2) for t in t_mean]
elps1_1 = 2*(2*h/(np.array(t1[:3]).mean()**2))/d
elps1_2= 2*(2*h/(np.array(t1[3:6]).mean()**2))/d
elps1_3 = 2*(2*h/(np.array(t1[6:9]).mean()**2))/d
elps1_4 = 2*(2*h/(np.array(t1[9:]).mean()**2))/d
elps2_1= 2*(2*h/(np.array(t2[:3]).mean()**2))/d
elps2_2= 2*(2*h/(np.array(t2[3:6]).mean()**2))/d
elps2_3 = 2*(2*h/(np.array(t2[6:9]).mean()**2))/d
elps2_4 = 2*(2*h/(np.array(t2[9:]).mean()**2))/d
elps3_1= 2*(2*h/(np.array(t3[:3]).mean()**2))/d
elps3_2= 2*(2*h/(np.array(t3[3:6]).mean()**2))/d
elps3_3 = 2*(2*h/(np.array(t3[6:9]).mean()**2))/d
elps3_4 = 2*(2*h/(np.array(t3[9:]).mean()**2))/d
elps4_1= 2*(2*h/(np.array(t4[:3]).mean()**2))/d
elps4_2= 2*(2*h/(np.array(t4[3:6]).mean()**2))/d
elps4_3 = 2*(2*h/(np.array(t4[6:9]).mean()**2))/d
elps4_4 = 2*(2*h/(np.array(t4[9:]).mean()**2))/d
elps5_1= 2*(2*h/(np.array(t5[:3]).mean()**2))/d
elps5_2= 2*(2*h/(np.array(t5[3:6]).mean()**2))/d
elps5_3 = 2*(2*h/(np.array(t5[6:9]).mean()**2))/d
elps5_4 = 2*(2*h/(np.array(t5[9:]).mean()**2))/d
elps6_1= 2*(2*h/(np.array(t6[:3]).mean()**2))/d
elps6_2= 2*(2*h/(np.array(t6[3:6]).mean()**2))/d
elps6_3 = 2*(2*h/(np.array(t6[6:9]).mean()**2))/d
elps6_4 = 2*(2*h/(np.array(t6[9:]).mean()**2))/d
elps1=[elps1_1, elps1_2, elps1_3, elps1_4]
elps2=[elps2_1, elps2_2, elps2_3, elps2_4]
elps3=[elps3_1, elps3_2, elps3_3, elps3_4]
elps4=[elps4_1, elps4_2, elps4_3, elps4_4]
elps5=[elps5_1, elps5_2, elps5_3, elps5_4]
elps6=[elps6_1, elps6_2, elps6_3, elps6_4]
els = []#[[elps1[i]]+[elps2[i]]+[elps3[i]]+[elps4[i]]+[elps5[i]]+[elps6[i]] for i in range(0, len(elps1))]
for i in range(0, len(elps1)):
    els.append(elps1[i])
    els.append(elps2[i])
    els.append(elps3[i])
    els.append(elps4[i])
    els.append(elps5[i])
    els.append(elps6[i])
# Функция для вычисления углового ускорения
def calculate_angular_acceleration(a_values, d):
    return [2 * a / d for a in a_values]
a_values = calculate_acceleration(h, t_mean)
# Функция для вычисления момента силы натяжения
def calculate_moment_of_force(m_karetki, m_shaibi, a_values, d, g=9.8):
    return [((m_karetki + m_shaibi[i%4]) *d*(g-a_values[i])/2) for i in range(0, len(a_values))]

print(calculate_moment_of_force(m_karetki, m_shaibi, a_values, d))
# Вычисление погрешности ускорения
def calculate_delta_a(a_values, t_mean):
    return [a * 2 * delta_t / t for a, t in zip(a_values, t_mean)]

# Вычисление погрешности углового ускорения
def calculate_delta_epsilon(delta_a_values, d):
   return [eps * np.sqrt((delta_a / a) ** 2 + (delta_d / d) ** 2) for eps, delta_a, a in zip(epsilon_values, delta_a_values, a_values)]

# Вычисление погрешности момента силы
def calculate_delta_m(delta_a_values, m_shaibi, m_karetki):
  return [M * np.sqrt((delta_m / (m_karetki + m_shaibi[i%4])) ** 2 + (delta_a / a) ** 2) for M, delta_a, a, i in zip(M_values, delta_a_values, a_values, (0, len(a_values)))]

# Вычисления
epsilon_values = calculate_angular_acceleration(a_values, d)
M_values = sorted(calculate_moment_of_force(m_karetki, m_shaibi, a_values, d))

# Погрешности
delta_a_values = calculate_delta_a(a_values, t_mean)
delta_epsilon_values = calculate_delta_epsilon(delta_a_values, d)
delta_M_values = calculate_delta_m(delta_a_values, m_shaibi, m_karetki)
#print(epsilon_values)


z = 1.96  # Для 95% доверительного интервала

# Первые значения a, ε и M
a_1 = a_values[0]
epsilon_1 = epsilon_values[0]
M_1 = M_values[0]

# Погрешности первых значений
delta_a_1 = delta_a_values[0]
delta_epsilon_1 = delta_epsilon_values[0]
delta_M_1 = delta_M_values[0]

# Вычисление доверительных интервалов
CI_a = (a_1 - z * delta_a_1, a_1 + z * delta_a_1)
CI_epsilon = (epsilon_1 - z * delta_epsilon_1, epsilon_1 + z * delta_epsilon_1)
CI_M = (M_1 - z * delta_M_1, M_1 + z * delta_M_1)

# Вывод доверительных интервалов
print(f"Доверительный интервал для a: {CI_a}")
print(f"Доверительный интервал для ε: {CI_epsilon}")
print(f"Доверительный интервал для M: {CI_M}")
#print(a_values[0], epsilon_values[0], M_values[0])

