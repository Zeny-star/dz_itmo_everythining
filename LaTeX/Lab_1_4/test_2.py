from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import AutoMinorLocator

# Функция для аппроксимации
def mapping(x, a, c):
    return a * x + c

# Функция для суммирования (может быть полезна в дальнейшем)
def summ(a):
    b = 0
    for i in range(len(a)):
        b += a[i]
    return b

# Исходные данные
Num = 6
#M = [0.0599, 0.0600, 0.0601, 0.0601, 0.0601, 0.0602,
#     0.1094, 0.1095, 0.1096, 0.1097, 0.1097, 0.1085,
#     0.1585, 0.1588, 0.1590, 0.1591, 0.1592, 0.1593,
#     0.2089, 0.2065, 0.2073, 0.2078, 0.2082, 0.2084]
M=[0.059755343782024906, 0.05988805356616932, 0.05996678191940275, 0.06002620346679002, 0.06003070832542,
0.06009519203935117, 0.1083239825796194, 0.10873704258924409, 0.10901011136271516, 0.10922532795,
0.10934837027680731, 0.1094349506628177, 0.15589310415431526, 0.15701414018702284, 0.157617362,
0.15818011754821498, 0.15847794981903882, 0.15862314260819682, 0.20327033775689562, 0.205055,
0.20624316249872993, 0.20689668118075552, 0.20727494174685787, 0.20761911348407774]
epsilon = [2.7497, 1.9375, 1.3600, 1.0293, 0.7231, 0.5979,
           5.3280, 3.8621, 2.8151, 2.0013, 1.6067, 1.2584,
           7.8761, 5.6579, 4.1388, 3.0464, 2.3494, 2.0122,
           9.2518, 7.0662, 5.3916, 4.2153, 3.0738, 2.4495]
#epsilon=[3.0193086947679486, 2.079723836442793, 1.522327340804521, 1.10162297041268, 1.0697285853220655, 0.6131840915928066, 
 #        5.612144181150749,4.008793511277738, 2.9488385636563086, 2.1134450328171046, 1.6358388932381425, 1.2997649168835927, 
  #       9.263818326817658, 6.26641982277470, 4.653537658107173, 3.148858302700897, 2.3525217203102615, 1.964308820526092,
   #      11.573529757565781, 7.933058585071003, 5.51127894170497, 4.178609004073315, 3.407251583236235, 2.705408865972692]
# Подготовка графиков и данных для аппроксимации
fig, ax = plt.subplots()
args = [[] for _ in range(Num)]
covar = [[] for _ in range(Num)]

x = [[] for _ in range(Num)]
y = [[] for _ in range(Num)]

# Заполнение x и y для каждого набора данных
for i in range(Num):
    for k in range(4):
        y[i].append(M[i + k * 6])
        x[i].append(epsilon[i + k * 6])

# Определение погрешностей
yerr = [[0.005] * 4 if i == 0 else [0.005] * 4 if i == 1 else [0.011] * 4 for i in range(Num)]
xerr = [[0.6] * 4 if i == 0 else [0.008] * 4 if i == 1 else [0.1] * 4 for i in range(Num)]

# Цвета и подписи для графиков
colors = ['r', 'b', 'g', 'y', 'k', 'm']
labels = ['1: ', '2: ', '3: ', '4: ', '5: ', '6: ']

# Построение графиков и расчет параметров аппроксимации
for i in range(Num):
    k_range = np.arange(min(x[i]) * 0.98, max(x[i]) * 1.1, 0.01)
    try:
        args[i], covar[i] = curve_fit(mapping, x[i], y[i])
        ax.plot(k_range, mapping(k_range, *args[i]), colors[i],
                label=labels[i] + r"$a = $" + str(round(args[i][0], 4)) +
                r", $b = $" + str(round(args[i][1], 4)) +
                r", e = " + str(round(sqrt(covar[i][0][0]) / args[i][0] * 100, 2)) + '%')
    except Exception as e:
        print(f"Error in curve fitting for dataset {i}: {e}")

# Добавление ошибок погрешности
ax.errorbar(x[0][0], y[0][0], yerr=yerr[0][0], xerr=xerr[0][0], color=colors[0][0], fmt='.', capsize=2)

# Настройки графика
ax.set_xlabel(r"$\epsilon, \frac{rad}{c^2}$", fontsize=14)
ax.set_ylabel(r'$M, \frac{kg \cdot m^2}{c^2}$', fontsize=14)
ax.legend(loc='best', prop={'size': 8})
ax.grid(which="major")
ax.grid(which="minor", linestyle=":")
ax.xaxis.set_minor_locator(AutoMinorLocator(10))
ax.yaxis.set_minor_locator(AutoMinorLocator(10))

# Отображение графика


# Рассчитаем сумму произведений, суммы квадратов и средние значения
#n = len(M)
#sum_M = np.sum(M)
#sum_epsilon = np.sum(epsilon)
#sum_M_epsilon = np.sum(np.array(M) * epsilon)
#sum_epsilon_squared = np.sum(np.array(epsilon) ** 2)
group_size = 4
num_groups = len(M) // group_size
I=[]
M_tr=[]
for i in range(num_groups):
    group_M = M[i * group_size:(i + 1) * group_size]
    group_epsilon = epsilon[i * group_size:(i + 1) * group_size]
    n = group_size
    sum_M = np.sum(group_M)
    sum_epsilon = np.sum(group_epsilon)
    sum_M_epsilon = np.sum(np.array(group_M) * group_epsilon)
    sum_epsilon_squared = np.sum(np.array(group_epsilon) ** 2)
    mean_M = np.mean(group_M)
    mean_epsilon = np.mean(group_epsilon)
    I.append((n * sum_M_epsilon - sum_M * sum_epsilon) / (n * sum_epsilon_squared - sum_epsilon ** 2))
    M_tr.append((sum_M - I[i] * sum_epsilon) / n)

I=sorted(I)

I=np.array([0.0237, 0.0289, 0.0374, 0.0474, 0.0639, 0.0782])
print(x, y)

# Значения в метрах
l1 = 0.057  # в метрах
l0 = 0.025  # в метрах
b = 0.040   # в метрах
# Данные для R и R^2
R = np.array([l1 + (i - 1) * l0 +  0.5* b for i in range(1, 7)])
R_squared = R ** 2
print(R_squared)
I_values = np.array(I)
M_tr_values = np.array(M_tr)
# Построение графика
# Построим экспериментальные точки зависимости I(R^2)
#print(f"Наклон (m): {m:.4f}")
sigma_l0 = sigma_l1 = sigma_b = 0.0005  # погрешности в метрах
sigma_M = 0.0005  # погрешность массы

# Вычисление R и R^2
R = np.array([l1 + (i - 1) * l0 + 0.5 * b for i in range(1, 7)])
R_squared = R ** 2

# Производные
dI_dl0 = np.gradient(R_squared, l0)
dI_dl1 = np.gradient(R_squared, l1)
dI_db = np.gradient(R_squared, b)

# Комбинированная погрешность для каждого положения утяжелителей
sigma_I = np.sqrt((dI_dl0 * sigma_l0) ** 2 + (dI_dl1 * sigma_l1) ** 2 + (dI_db * sigma_b) ** 2 + (sigma_M / epsilon[0])**2)

# Вывод погрешности для каждого положения
# Построение графика
#plt.scatter(x, y, color='red', label='Данные')
#plt.plot(x, m*x + I, 'b-', label='I(R^2)')
#plt.xlabel('R^2, m^2')
#plt.ylabel('I, kgm^2')
#plt.legend()
#plt.grid()
plt.show()
# Теперь построим линейные зависимости для каждого положения утяжелителей
epsilon_range = np.linspace(min(epsilon), max(epsilon), 100)

for i, (I, M_tr) in enumerate(zip(I_values, M_tr_values)):
    M_fitted = M_tr + I * epsilon_range
    plt.plot(epsilon_range, M_fitted, label=f'Группа {i+1}: I={I:.4f}, Mтр={M_tr:.4f}')
# Оформление графика
plt.xlabel(r'$\epsilon$ (угловое ускорение), $\frac{1}{c^2}$', fontsize=12)
plt.ylabel('Момент силы Трения, H', fontsize=12)
plt.title('Зависимость Mтр(ε) и экспериментальные точки', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
print(f"Момент инерции I: {I}")
print(f"Момент силы трения Mтр: {M_tr}")
print(f"Коэффициент детерминированности R_squared: {R_squared}")
print('Погрешность для каждого положения утяжелителей:', sigma_I)



