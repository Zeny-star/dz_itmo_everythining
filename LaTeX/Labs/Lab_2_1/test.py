import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Инициализация стиля графиков
plt.style.use('seaborn')
colors = ['#5986e4', '#7d4dc8', '#b01bb3']

# Константы
p0 = 1e5  # Атмосферное давление, Па
mmHg_to_Pa = 13.55 * 9.80665  # Коэффициент перевода мм рт.ст. в Па

# Исходные данные (в мм рт.ст.)
data = {
    't1': {
        'temp': 21.5 + 273.15,
        'p1': [35.6, 14.6, 2.9, -9.2, -27.7, -38.9, -48.1, -52.1],
        'p2': [35.6, 0.3, -13, -25.2, -33.7, -41.5, -47.3, -52.1]
    },
    't2': {
        'temp': 31.2 + 273.15,
        'p1': [18.2, 1.9, -12.1, -23.3, -32, -39.6, -51.3, -55.1],
        'p2': [12.9, -6.1, -18.6, -28.9, -37.4, -44.9, -50.2, -55.1]
    },
    't3': {
        'temp': 40.1 + 273.15,
        'p1': [12.6, -4.2, -15.9, -27.9, -35.5, -42.5, -49.3, -53.2],
        'p2': [12.5, -1.3, -15.2, -27, -35.4, -42.8, -48.7, -53.2]
    },
    't4': {
        'temp': 49.0 + 273.15,
        'p1': [15.4, -1.7, -14.7, -25.4, -34.8, -41.4, -50.8, -55.3],
        'p2': [11.7, -3.9, -18.5, -28.6, -37.2, -44.7, -50, -55.4]
    },
    't5': {
        'temp': 60.0 + 273.15,
        'p1': [13.6, -4.5, -15.5, -27.6, -36.2, -42.4, -49, -53.3],
        'p2': [13.6, -0.2, -16.7, -26.5, -35.5, -42.4, -48.1, -53.3]
    }
}

# Объемы цилиндров (м³)
V_cyl = np.array([50, 60, 70, 80, 90, 100, 110, 120]) * 1e-6

# Преобразование данных в массивы
temps = np.array([data[key]['temp'] for key in data])
pressures = {}

for key in data:
    p1 = np.array(data[key]['p1']) * mmHg_to_Pa
    p2 = np.array(data[key]['p2']) * mmHg_to_Pa
    pressures[key] = p0 + (p1 + p2)/2

# Расчет коэффициентов a и c для каждого объема
a_list, c_list = [], []
delta_a_list, delta_c_list = [], []
t_star_list, delta_t_star_list = [], []

for i in range(len(V_cyl)):
    p = np.array([pressures[key][i] for key in data])
    
    # Линейная регрессия p = a*T + c
    slope, intercept, r_value, p_value, std_err = stats.linregress(temps, p)
    
    a_list.append(slope)
    c_list.append(intercept)
    
    # Погрешности коэффициентов
    n = len(temps)
    S_xx = np.sum((temps - np.mean(temps))**2)
    delta_a = std_err * np.sqrt(n / S_xx)
    delta_c = std_err * np.sqrt(1/n + np.mean(temps)**2 / S_xx)
    
    delta_a_list.append(delta_a)
    delta_c_list.append(delta_c)
    
    # Расчет t* и его погрешности
    t_star = -intercept / slope
    delta_t_star = t_star * np.sqrt((delta_a/slope)**2 + (delta_c/intercept)**2)
    
    t_star_list.append(t_star)
    delta_t_star_list.append(delta_t_star)

# Построение графиков p(T) для V = 50, 90, 120 мл
selected_volumes = [0, 4, 7]
plt.figure(figsize=(10, 6))
for idx in selected_volumes:
    V = V_cyl[idx] * 1e6
    p = [pressures[key][idx] for key in data]
    plt.scatter(temps, p, label=f'V = {V:.0f} мл')
    
    # Линия регрессии
    T_range = np.linspace(min(temps), max(temps), 100)
    plt.plot(T_range, a_list[idx]*T_range + c_list[idx], '--')

plt.xlabel('Температура, K')
plt.ylabel('Давление, Па')
plt.legend()
plt.grid(True)
plt.title('Зависимость давления от температуры')
plt.show()

# Зависимость t* от 1/V
inv_V = 1 / V_cyl

# Линейная регрессия t* = A'*(1/V) + C'
slope, intercept, r_value, p_value, std_err = stats.linregress(inv_V, t_star_list)

# Погрешности коэффициентов
n = len(inv_V)
S_xx = np.sum((inv_V - np.mean(inv_V))**2)
delta_A_prime = std_err * np.sqrt(n / S_xx)
delta_C_prime = std_err * np.sqrt(1/n + np.mean(inv_V)**2 / S_xx)

print(f'Оценка абсолютного нуля: {intercept:.2f} ± {delta_C_prime:.2f} K')

# График t*(1/V)
plt.figure(figsize=(10, 6))
plt.scatter(inv_V, t_star_list, label='Экспериментальные точки')
plt.plot(inv_V, slope*inv_V + intercept, 'r-', label=f'Линейная аппроксимация: $t^* = {slope:.2f}(1/V) + {intercept:.2f}$')
plt.xlabel('$1/V$, 1/м³')
plt.ylabel('$t^*$, K')
plt.legend()
plt.grid(True)
plt.title('Экстраполяция к абсолютному нулю')
plt.show()
