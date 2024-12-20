
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Константы
w = 7.2921e-5  # угловая скорость вращения Земли
R_0 = 6371e3  # радиус Земли в метрах
G = 6.67e-11  # гравитационная постоянная
M_e = 5.97e24  # масса Земли в кг

# Дифференциальные уравнения
def equations(t, state, R, M, pxi_0):
    x, y, z, vx, vy, vz = state
    g_E0 = G * M / R**2
    alpha = w**2 * R / g_E0 * np.sin(pxi_0) * np.cos(pxi_0)
    teta = pxi_0 - alpha
    w_x = -w * np.sin(teta)
    w_z = w * np.cos(teta)
    g_0 = g_E0 * (1 - w**2 * R / g_E0 * np.sin(pxi_0)**2)
    g_x_x = -(g_E0 / R - w_z**2)
    g_y_y = -(g_E0 / R - w**2)
    g_z_z = 2 * g_E0 / R + w_x**2
    g_x_z = 3 * alpha * g_E0 / R - w_z * w_x

    ax_1 = g_x_x * x + g_x_z * z + 2 * w_z * vy
    ay_1 = g_y_y * y - 2 * w_z * vx + 2 * w_x * vz
    az_1 = -g_0 + g_x_z * x + g_z_z * z - 2 * w_x * vy

    return [vx, vy, vz, ax_1, ay_1, az_1]

# Вычисление координат
def calculate_final_coordinates(initial_height, initial_angle, initial_velocity, planet_radius, planet_mass):
    pxi_0 = np.radians(initial_angle)
    R = planet_radius
    M = planet_mass

    # Начальные условия
    initial_state = [0, 0, initial_height, 0, initial_velocity, 0]
    g_E0 = G * M / R**2
    g_0 = g_E0 * (1 - w**2 * R / g_E0 * np.sin(pxi_0)**2)

    # Временной интервал
    t_span = (0, (2 * initial_state[2] / g_0)**0.5)
    t_eval = np.linspace(*t_span, 1000)

    # Решение дифференциальных уравнений
    solution = solve_ivp(equations, t_span, initial_state, t_eval=t_eval, method='RK45', args=(R, M, pxi_0))

    # Конечные координаты
    x_final, y_final = solution.y[0, -1], solution.y[1, -1]
    return x_final, y_final

# Построение графиков
def update_plot(val):
    initial_angle = angle_slider.val
    initial_velocity = velocity_slider.val
    planet_radius = radius_slider.val
    planet_mass = mass_slider.val

    initial_heights = np.linspace(4, 1596, 100)
    final_x = []
    final_y = []

    for h in initial_heights:
        x, y = calculate_final_coordinates(h, initial_angle, initial_velocity, planet_radius, planet_mass)
        final_x.append(x)
        final_y.append(y)

    ax1.clear()
    ax2.clear()

    ax1.plot(initial_heights, final_x, label="Конечный X", color="blue")
    ax1.set_xlabel("Начальная высота (м)")
    ax1.set_ylabel("Конечный X (м)")
    ax1.set_title("Зависимость конечного X от начальной высоты")
    ax1.grid()
    ax1.legend()

    ax2.plot(initial_heights, final_y, label="Конечный Y", color="green")
    ax2.set_xlabel("Начальная высота (м)")
    ax2.set_ylabel("Конечный Y (м)")
    ax2.set_title("Зависимость конечного Y от начальной высоты")
    ax2.grid()
    ax2.legend()

    plt.draw()

# Создание окна
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(left=0.25, right=0.95, bottom=0.25)

# Ползунки
ax_angle = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_velocity = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_radius = plt.axes([0.25, 0.05, 0.65, 0.03])
ax_mass = plt.axes([0.25, 0.0, 0.65, 0.03])

angle_slider = Slider(ax_angle, 'Угол (градусы)', 0, 90, valinit=45, valstep=1)
velocity_slider = Slider(ax_velocity, 'Скорость_y (м/с)', 0, 1000, valinit=0, valstep=10)
radius_slider = Slider(ax_radius, 'Радиус (м)', 1e6, 1e7, valinit=R_0, valstep=1e5)
mass_slider = Slider(ax_mass, 'Масса (кг)', 1e23, 1e25, valinit=M_e, valstep=1e23)

# Подключение обновления графика к ползункам
angle_slider.on_changed(update_plot)
velocity_slider.on_changed(update_plot)
radius_slider.on_changed(update_plot)
mass_slider.on_changed(update_plot)

# Инициализация графиков
update_plot(None)

plt.show()

