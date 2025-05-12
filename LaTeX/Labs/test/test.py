import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# Настройки для отображения в стиле LaTeX (если LaTeX установлен)
try:
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman'] # Или 'Times New Roman', 'DejaVu Serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
except RuntimeError:
    print("LaTeX не найден, используется стандартный рендеринг Matplotlib.")
    plt.rcParams['mathtext.fontset'] = 'dejavusans' # запасной вариант для мат. символов

# Цвета, приближенные к изображению
color_t1 = '#2E3B4E'
color_t2 = '#77B5FE'
color_t3 = '#FDB813' # Желто-оранжевый

# ----- Генерация данных для кривых (аппроксимация) -----

# t1: Гладкая кривая, проходящая через ключевые точки
x_t1_pts = np.array([0, 18.4, 53.1, 63.4, 90])
y_t1_pts = np.array([0.45, 0.6, 1.0, 1.1, 1.18])
cs_t1 = CubicSpline(x_t1_pts, y_t1_pts)
alpha_range_t1 = np.linspace(0, 90, 300)
t1_y = cs_t1(alpha_range_t1)

# t2: Начинается в alpha=63.4, убывает
alpha_range_t2 = np.linspace(63.4, 90, 100)
y_start_t2 = 0.38  # Начальное значение t2 при alpha=63.4
y_asymptote_t2 = 0.29 # Асимптота
t2_y = y_asymptote_t2 + (y_start_t2 - y_asymptote_t2) * np.exp(-0.1 * (alpha_range_t2 - 63.4))

# t3:
# Часть 1: Проходит через (0, 0.6), (18.4, 0.6), (53.1, 1.0)
x_t3_part1_pts = np.array([0, 18.4, 53.1])
y_t3_part1_pts = np.array([0.6, 0.6, 1.0])
# bc_type для управления наклонами на концах для лучшего соответствия
cs_t3_part1 = CubicSpline(x_t3_part1_pts, y_t3_part1_pts, bc_type=((1, 0.005), (2, 0)))
alpha_range_t3_part1 = np.linspace(0, 53.1, 100)
t3_y_part1 = cs_t3_part1(alpha_range_t3_part1)

# Часть 2: Экспоненциальный рост от (53.1, 1.0)
alpha_range_t3_part2 = np.linspace(53.1, 62.5, 50) # Обрезаем, чтобы остаться в пределах графика по y
t3_y_part2 = 1.0 * np.exp(0.12 * (alpha_range_t3_part2 - 53.1))

alpha_range_t3 = np.concatenate((alpha_range_t3_part1, alpha_range_t3_part2))
t3_y = np.concatenate((t3_y_part1, t3_y_part2))


# ----- Создание графика -----
fig, ax = plt.subplots(figsize=(6.5, 4.8)) # Размер графика

# Отрисовка кривых
ax.plot(alpha_range_t1, t1_y, label=r'$t_1$', color=color_t1, linewidth=1.5, zorder=2)
ax.plot(alpha_range_t2, t2_y, label=r'$t_2$', color=color_t2, linewidth=1.5, zorder=2)
ax.plot(alpha_range_t3, t3_y, label=r'$t_3$', color=color_t3, linewidth=1.5, zorder=3) # t3 поверх t1

# Подписи осей
ax.set_xlabel(r'$\alpha, 1^{\circ}$', fontsize=11)
ax.set_ylabel(r'$t, \mathrm{c}$', fontsize=11, rotation=0, ha='right', va='center')

# Позиционирование подписей осей (координаты в долях от размера осей)
ax.xaxis.set_label_coords(1.0, -0.085)
ax.yaxis.set_label_coords(-0.04, 1.03)


# Деления на осях (тики)
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_yticks([0.5, 1.0, 1.5])
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=10, direction='in', width=0.75)
ax.tick_params(axis='both', which='minor', direction='in', width=0.5)
ax.tick_params(axis='x', pad=5) # Отступ для подписей тиков по X
ax.tick_params(axis='y', pad=3) # Отступ для подписей тиков по Y

# Пределы осей (немного расширены для стрелок и подписей)
ax.set_xlim(-4, 92)
ax.set_ylim(0.3, 1.85)

# ----- Стилизация осей (spines) и стрелки -----
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(0.75)
ax.spines['left'].set_position(('data', -4)) # Сдвиг левой оси к x=-4
ax.spines['bottom'].set_linewidth(0.75)
ax.spines['bottom'].set_position(('data', 0.3)) # Сдвиг нижней оси к y=0.3

# Стрелки на концах осей
# Координаты концов осей для стрелок
xmin_plot, xmax_plot = ax.get_xlim()
ymin_plot, ymax_plot = ax.get_ylim()
y_pos_x_axis = 0.3 # y-координата горизонтальной оси
x_pos_y_axis = -4  # x-координата вертикальной оси

# X-ось стрелка
ax.plot(xmax_plot, y_pos_x_axis, ">k", markersize=5, clip_on=False, zorder=5)
# Y-ось стрелка
ax.plot(x_pos_y_axis, ymax_plot, "^k", markersize=5, clip_on=False, zorder=5)


# ----- Аннотации -----
# Координаты точек для аннотаций
pt1_x, pt1_y = 18.4, cs_t1(18.4)      # Должно быть (18.4, 0.6)
pt2_x, pt2_y = 53.1, cs_t1(53.1)      # Должно быть (53.1, 1.0)
pt3_x = 63.4
pt3_y_t1 = cs_t1(pt3_x)               # Должно быть (63.4, 1.1)
pt3_y_t2 = t2_y[0]                    # Должно быть (63.4, 0.38)

# Отрисовка точек
ax.plot([pt1_x, pt2_x, pt3_x], [pt1_y, pt2_y, pt3_y_t1], 'o', color='black', markersize=3, zorder=4)
ax.plot(pt3_x, pt3_y_t2, 'o', color='black', markersize=3, zorder=4)

# Пунктирная линия
ax.plot([pt3_x, pt3_x], [pt3_y_t2, pt3_y_t1], '--', color='black', linewidth=0.7, zorder=1)

# Текстовые аннотации
ax.text(pt1_x, pt1_y - 0.14, r'$\alpha = 18.4^{\circ}$', ha='center', va='top', fontsize=10, zorder=5)
ax.text(pt2_x - 2, pt2_y + 0.06, r'$\alpha = 53.1^{\circ}$', ha='right', va='bottom', fontsize=10, zorder=5)
ax.text(pt3_x + 2.5, (pt3_y_t1 + pt3_y_t2) / 2, r'$\alpha = 63.4^{\circ}$',
        ha='left', va='center', fontsize=10, zorder=5)

# ----- Легенда -----
legend = ax.legend(loc='upper right', frameon=True, fontsize=9.5, borderpad=0.5)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(0.6)
legend.set_zorder(10) # Легенда поверх всего

# Убираем лишние отступы
plt.tight_layout(pad=0.3)
# Иногда требуется ручная корректировка отступов, если tight_layout не справляется
# fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

plt.show()

