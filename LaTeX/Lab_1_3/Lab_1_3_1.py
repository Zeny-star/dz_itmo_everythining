import numpy as np

# Функция для расчета средних значений, погрешностей и доверительных интервалов
def calc_conf_intervals(delta_p, delta_W, N, t_alpha_N):
    # Средние значения
    delta_p_mean = np.mean(delta_p)
    delta_W_mean = np.mean(delta_W)

    # Стандартные отклонения
    sigma_delta_p = np.sqrt(np.sum((delta_p - delta_p_mean)**2) / (N - 1))
    sigma_delta_W = np.sqrt(np.sum((delta_W - delta_W_mean)**2) / (N - 1))

    # Погрешности средних значений
    Delta_delta_p = t_alpha_N * sigma_delta_p / np.sqrt(N)
    Delta_delta_W = t_alpha_N * sigma_delta_W / np.sqrt(N)

    # Доверительные интервалы
    conf_interval_delta_p = (delta_p_mean - Delta_delta_p, delta_p_mean + Delta_delta_p)
    conf_interval_delta_W= (delta_W_mean - Delta_delta_W, delta_W_mean + Delta_delta_W)

    return (delta_p_mean, Delta_delta_p, conf_interval_delta_p), (delta_W_mean, Delta_delta_W, conf_interval_delta_W)

# Функция для расчета импульсов
def calc_impulses(m1, m2, v_10, v1, v2):
    # Импульсы до и после соударения
    p_10x = m1 * v_10
    p1x = m1 * v1
    p2x = m2 * v2

    # Относительные изменения импульса и энергии
    delta_p = (p1x + p2x) / p_10x - 1
    delta_W = (m1 * v1**2 + m2 * v2**2) / (m1 * v_10**2) - 1

    return delta_p, delta_W
def cacl_th(m1, m2):
    return -m2/(m1 + m2)
# Данные измерений
# Для первой таблицы
v_10_1 = np.array([0.49, 0.52, 0.51, 0.51, 0.51])
v_1_1 = np.array([0.06, 0.06, 0.07, 0.07, 0.07])
v_2_1 = np.array([0.44, 0.47, 0.46, 0.46, 0.46])
m_1_1 = 0.04964
m_2_1 = 0.05

# Для второй таблицы
v_10_2 = np.array([0.53, 0.53, 0.53, 0.52, 0.52])
v_1_2 = np.array([-0.05, -0.05, -0.05, -0.05, -0.07])
v_2_2 = np.array([0.31, 0.30, 0.28, 0.31, 0.27])
m_2_2 = 0.04964
m_1_2 = 0.1003

# Параметры
N = 5  # количество измерений
t_alpha_N = 2.78  # коэффициент Стьюдента для N=5 и доверительной вероятности 0.95

# Расчеты для первой таблицы
delta_p_1, delta_W_1 = calc_impulses(m_1_1, m_2_1, v_10_1, v_1_1, v_2_1)
(delta_p_mean_1, Delta_delta_p_1, conf_interval_delta_p_1), (delta_W_mean_1, Delta_delta_W_1, conf_interval_delta_W_1) = calc_conf_intervals(delta_p_1, delta_W_1, N, t_alpha_N)

# Расчеты для второй таблицы
delta_p_2, delta_W_2 = calc_impulses(m_1_2, m_2_2, v_10_2, v_1_2, v_2_2)
(delta_p_mean_2, Delta_delta_p_2, conf_interval_delta_p_2), (delta_W_mean_2, Delta_delta_W_2, conf_interval_delta_W_2) = calc_conf_intervals(delta_p_2, delta_W_2, N, t_alpha_N)
# Вывод результатов
print("Результаты для первой таблицы:")
print(f"Среднее значение относительного изменения импульса: {delta_p_mean_1:.5f}, погрешность: {Delta_delta_p_1:.5f}, доверительный интервал: {conf_interval_delta_p_1}")
print(f"Среднее значение относительного изменения энергии: {delta_W_mean_1:.5f}, погрешность: {Delta_delta_W_1:.5f}, доверительный интервал: {conf_interval_delta_W_1}")
print('delta_delta_p для 1:', Delta_delta_p_1)
print('delta_delta_W для 1:', Delta_delta_W_1)
print('Th значение:', cacl_th(m_1_1, m_2_1))

print("\nРезультаты для второй таблицы:")
print(f"Среднее значение относительного изменения импульса: {delta_p_mean_2:.5f}, погрешность: {Delta_delta_p_2:.5f}, доверительный интервал: {conf_interval_delta_p_2}")
print(f"Среднее значение относительного изменения энергии: {delta_W_mean_2:.5f}, погрешность: {Delta_delta_W_2:.5f}, доверительный интервал: {conf_interval_delta_W_2}")
print('delta_delta_p для 2:', Delta_delta_p_2)
print('delta_delta_W для 2:', Delta_delta_W_2)
print('Th значение:', cacl_th(m_1_2, m_2_2))
