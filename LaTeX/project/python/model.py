import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib

# Константы
w = 7.2921e-5
R_0 = 6371e3
G = 6.67e-11
M_e = 5.97e24
M_sun = 1.989e30


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

# Вычисление конечных координат
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

# Обновление графиков
def update_plot(val):
    initial_angle = angle_slider.val
    initial_velocity_x = velocity_x_slider.val
    initial_velocity_y = velocity_y_slider.val
    initial_velocity_z = velocity_z_slider.val
    planet_radius = radius_slider.val
    planet_mass = mass_slider.val

    initial_heights = np.linspace(4, 1596, 100)
    final_x = []
    final_y = []
    approximations = []
    differences = []

    for h in initial_heights:
        x, y = calculate_final_coordinates(h, initial_angle, initial_velocity_x, initial_velocity_y, initial_velocity_z, planet_radius, planet_mass)
        final_x.append(x)
        final_y.append(y)
        approximation = np.sqrt(x**2 + y**2)  # Аппроксимация
        approximations.append(approximation)
        differences.append(approximation - y)  # Разность между аппроксимацией и Y

    # Очистка предыдущих графиков
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # График 1: Конечный X
    ax1.plot(initial_heights, final_x, label="$\Delta x$", color="blue")
    ax1.set_xlabel("$h (м)$")
    ax1.set_ylabel("Конечный $x (м)$")
    ax1.legend()

    # График 2: Конечный Y
    ax2.plot(initial_heights, final_y, label="$\Delta y$", color="green")
    ax2.plot(initial_heights, approximations, label="Аппроксимация", color="red")
    ax2.set_xlabel("Начальная высота (м)")
    ax2.set_ylabel("Аппроксимация (м)")
    ax2.set_title("Аппроксимация расстояния")
    ax2.legend()

    # График 4: Разность аппроксимации и Y
    ax3.plot(initial_heights, differences, label="Разность (Аппр - $\Delta y$)", color="purple")
    ax3.set_xlabel("Начальная высота $(м)$")
    ax3.set_ylabel("Разность $(м)$")
    ax3.set_title("Разность аппроксимации и $y$")
    ax3.legend()

    plt.draw()

# Настройка фигуры
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(left=0.25, right=0.95, bottom=0.35, hspace=0.4, wspace=0.4)

# Ползунки
ax_angle = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_velocity_x = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_velocity_y = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_velocity_z = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_radius = plt.axes([0.25, 0.05, 0.65, 0.03])
ax_mass = plt.axes([0.25, 0.0, 0.65, 0.03])

angle_slider = Slider(ax_angle, r'$\psi_0^{\circ}$', 0, 90, valinit=45, valstep=1)
velocity_x_slider = Slider(ax_velocity_x, '$v_{0x}$ (м/с)', 0, 1000, valinit=0, valstep=0.1)
velocity_y_slider = Slider(ax_velocity_y, '$v_{0y}$ (м/с)', 0, 1000, valinit=0, valstep=0.1)
velocity_z_slider = Slider(ax_velocity_z, '$v_{oz}$ (м/с)', 0, 1000, valinit=0, valstep=0.1)
radius_slider = Slider(ax_radius, '$R_E$ (м)', 1e6, 1e7, valinit=R_0, valstep=1e5)
mass_slider = Slider(ax_mass, '$M_{E}$ (кг)', M_e, M_sun, valinit=M_e, valstep=(M_sun - M_e) / 100)

# Привязка обновления графиков к ползункам
angle_slider.on_changed(update_plot)
velocity_x_slider.on_changed(update_plot)
velocity_y_slider.on_changed(update_plot)
velocity_z_slider.on_changed(update_plot)
radius_slider.on_changed(update_plot)
mass_slider.on_changed(update_plot)

# Инициализация графиков
update_plot(None)

plt.show()

