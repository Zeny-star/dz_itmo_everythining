import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams.update({'font.size': 15})
w = 7.2921e-5
R_0 = 6371e3
G = 6.67e-11
M_e = 5.97e24
M_sun = 1.989e30

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

def calculate_final_coordinates(initial_height, initial_angle, initial_velocity_x, initial_velocity_y, initial_velocity_z, planet_radius, planet_mass):
    pxi_0 = np.radians(initial_angle)
    R = planet_radius
    M = planet_mass
    initial_state = [0, 0, initial_height, initial_velocity_x, initial_velocity_y, initial_velocity_z]
    g_E0 = G * M / R**2
    g_0 = g_E0 * (1 - w**2 * R / g_E0 * np.sin(pxi_0)**2)
    t_span = (0, (2 * initial_state[2] / g_0)**0.5)
    t_eval = np.linspace(*t_span, 1000)
    solution = solve_ivp(equations, t_span, initial_state, t_eval=t_eval, method='DOP853', args=(R, M, pxi_0))
    x_final, y_final = solution.y[0, -1], solution.y[1, -1]
    return x_final, y_final

def update_plot(val):
    initial_angle_fixed = 45  # Фиксированный угол 45 градусов
    velocities = [
        (0.1, 0),  # vx = 0.1, vy = 0
        (0, 0.1),  # vx = 0, vy = 0.1
        (0, 0)     # vx = 0, vy = 0
    ]
    colors = ['blue', 'green', 'red']
    labels = ['$v_x = 0.1$, $v_y = 0$', '$v_x = 0$, $v_y = 0.1$', '$v_x = 0$, $v_y = 0$']

    ax.clear()  # Очищаем график перед обновлением

    for i, (vx, vy) in enumerate(velocities):
        final_displacement = []
        for h in initial_heights:
            x, y = calculate_final_coordinates(
                h, initial_angle_fixed, vx, vy, initial_velocity_z, planet_radius, planet_mass
            )
            # Вычисляем полное смещение как sqrt(x^2 + y^2)
            displacement = np.sqrt(x)
            final_displacement.append(displacement)
        # Добавляем точку (0, 0) в начало данных
        final_displacement = np.array(final_displacement)
        line, = ax.plot(np.insert(initial_heights, 0, 0), np.insert(final_displacement, 0, 0), color=colors[i], label=labels[i])

    ax.set_xlabel("$h$ (м)")
    ax.set_ylabel("Полное смещение (м)")
    ax.set_xlim(0, max(initial_heights))
    ax.set_ylim(0, None)
    ax.legend()
    plt.draw()

# Настройка фигуры и графика
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.25, top=0.9)

# Начальные параметры
initial_heights = np.linspace(4, 1596, 100)
initial_velocity_z = 0
planet_radius = R_0
planet_mass = M_e

# Слайдеры (скрыты по умолчанию)
ax_velocity_x = plt.axes([0.25, 0.1, 0.65, 0.03], visible=False)
ax_velocity_z = plt.axes([0.25, 0.05, 0.65, 0.03], visible=False)
ax_radius = plt.axes([0.25, 0.0, 0.65, 0.03], visible=False)

velocity_x_slider = Slider(ax_velocity_x, '$v_{0x}$ (м/с)', 0, 110, valinit=0.1, valstep=0.1)
velocity_z_slider = Slider(ax_velocity_z, '$v_{oz}$ (м/с)', 0, 1000, valinit=initial_velocity_z, valstep=0.1)
radius_slider = Slider(ax_radius, '$R_E$ (м)', 1e6, 1e7, valinit=planet_radius, valstep=1e5)

# Обновение графика при изменении слайдеров
velocity_x_slider.on_changed(update_plot)
velocity_z_slider.on_changed(update_plot)
radius_slider.on_changed(update_plot)

# Первоначальное построение графика
update_plot(None)

plt.show()
