import numpy as np
import matplotlib.pyplot as plt

# Данные момента силы и углового ускорения
M = np.array([0.059755343782024906, 0.05988805356616932, 0.05996678191940275, 0.06002620346679002, 
              0.06003070832542, 0.06009519203935117, 0.1083239825796194, 0.10873704258924409, 
              0.10901011136271516, 0.10922532795, 0.10934837027680731, 0.1094349506628177, 
              0.15589310415431526, 0.15701414018702284, 0.157617362, 0.15818011754821498, 
              0.15847794981903882, 0.15862314260819682, 0.20327033775689562, 0.205055, 
              0.20624316249872993, 0.20689668118075552, 0.20727494174685787, 0.20761911348407774])

epsilon = np.array([3.0193086947679486, 2.079723836442793, 1.522327340804521, 1.10162297041268, 
                    1.0697285853220655, 0.6131840915928066, 5.612144181150749, 4.008793511277738, 
                    2.9488385636563086, 2.1134450328171046, 1.6358388932381425, 1.2997649168835927, 
                    9.263818326817658, 6.26641982277470, 4.653537658107173, 3.148858302700897, 
                    2.3525217203102615, 1.964308820526092, 11.573529757565781, 7.933058585071003, 
                    5.51127894170497, 4.178609004073315, 3.407251583236235, 2.705408865972692])

# Значения в метрах
l1 = 0.057  # в метрах
l0 = 0.025  # в метрах
b = 0.040   # в метрах

# Данные для R и R^2
R = np.array([l1 + (n - 1) * l0 + 12 * b / 1000 for n in range(1, 7)])
R_squared = R ** 2

# Функция для расчета I и Mтр по МНК для каждой группы
def calculate_I_Mtr(M_group, epsilon_group):
    n = len(M_group)
    sum_M = np.sum(M_group)
    sum_epsilon = np.sum(epsilon_group)
    sum_M_epsilon = np.sum(M_group * epsilon_group)
    sum_epsilon_squared = np.sum(epsilon_group ** 2)
    
    # Момент инерции I (наклон прямой)
    I = (n * sum_M_epsilon - sum_M * sum_epsilon) / (n * sum_epsilon_squared - sum_epsilon ** 2)
    
    # Момент силы трения Mтр (свободный член)
    M_tr = (sum_M - I * sum_epsilon) / n
    
    return I, M_tr

# Разделим данные на группы (по 6 измерений для каждого положения утяжелителей)
group_size = 6
num_groups = len(M) // group_size

I_values = []
M_tr_values = []

# Рассчитываем I и Mтр для каждой группы
for i in range(num_groups):
    M_group = M[i * group_size : (i + 1) * group_size]
    epsilon_group = epsilon[i * group_size : (i + 1) * group_size]
    
    I, M_tr = calculate_I_Mtr(M_group, epsilon_group)
    I_values.append(I)
    print(I_values)
    M_tr_values.append(M_tr)

# Преобразуем I_values в массив
I_values = np.array(I_values)

# Проверяем соответствие размеров R_squared и I_values
if len(R_squared) != len(I_values):
    print(f"Ошибка: R_squared имеет длину {len(R_squared)}, а I_values — {len(I_values)}")
else:
    # Построение графика
    plt.figure(figsize=(10, 6))

    # Построим экспериментальные точки зависимости I(R^2)
    plt.scatter(R_squared, I_values, color='blue', label='Экспериментальные точки I(R^2)', zorder=5)

    # Теперь построим линейные зависимости для каждого положения утяжелителей
    epsilon_range = np.linspace(min(epsilon), max(epsilon), 100)

    for i, (I, M_tr) in enumerate(zip(I_values, M_tr_values)):
        M_fitted = M_tr + I * epsilon_range
        plt.plot(epsilon_range, M_fitted, label=f'Группа {i+1}: I={I:.4f}, Mтр={M_tr:.4f}')

    # Оформление графика
    plt.xlabel(r'$\epsilon$ (угловое ускорение)', fontsize=12)
    plt.ylabel('Момент силы M', fontsize=12)
    plt.title('Зависимость M(ε) и экспериментальные точки', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()
