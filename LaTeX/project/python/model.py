import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.widgets import Button

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


def equations_approx(t, state, R, M, pxi_0):
    x, y, z, vx, vy, vz = state
    g_E0 = G * M / R**2
    alpha = w**2 * R / g_E0 * np.sin(pxi_0) * np.cos(pxi_0)
    teta = pxi_0 - alpha
    w_x = -w * np.sin(teta)
    w_z = w * np.cos(teta)
    g_0 = g_E0 * (1 - w**2 * R / g_E0 * np.sin(pxi_0)**2)
    g_x_z = 3 * alpha * g_E0 / R - w_z * w_x
    ax = g_x_z*z+2*w_z*vy
    ay = 2*w_x*vz
    az = -g_0
    return [vx, vy, vz, ax, ay, az]

def calculate_final_coordinates_approx(initial_height, initial_angle, initial_velocity_x, initial_velocity_y, initial_velocity_z, planet_radius, planet_mass):
    # Преобразование угла в радианы
    pxi_0 = np.radians(initial_angle)
    R = planet_radius
    M = planet_mass

    # Начальное состояние: [x, y, z, vx, vy, vz]
    initial_state = [0, 0, initial_height, initial_velocity_x, initial_velocity_y, initial_velocity_z]

    # Функция события: остановка интеграции, когда z = 0
    def hit_ground(t, state):
        z = state[2]
        return z

    hit_ground.terminal = True  # Остановить интеграцию, когда z = 0
    hit_ground.direction = -1  # Искать только пересечение в отрицательном направлении (сверху вниз)

    # Диапазон времени
    t_span = (0, 1e6)  # Достаточно большой временной интервал для поиска

    # Решение уравнений
    solution = solve_ivp(
        lambda t, state: equations_approx(t, state, R, M, pxi_0),  # Используем lambda для передачи параметров
        t_span,
        initial_state,
        method='DOP853',
        events=hit_ground
    )
    
    # Извлечение конечных координат x, y
    x_final, y_final = solution.y[0, -1], solution.y[1, -1]
    return x_final, y_final

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
    differences_x = []

    for h in initial_heights:
        x, y = calculate_final_coordinates(
            h, initial_angle, initial_velocity_x, initial_velocity_y, initial_velocity_z, planet_radius, planet_mass
        )
        x_app, y_app = calculate_final_coordinates_approx(
            h, initial_angle, initial_velocity_x, initial_velocity_y, initial_velocity_z, planet_radius, planet_mass
        )
        final_x.append(x)
        final_y.append(y)
        approximations.append(y_app)  # Добавляем значение аппроксимации в список
        differences.append(y_app - y)  # Разность между реальным значением и аппроксимацией Y
        differences_x.append(x_app - x)
    # Очистка предыдущих графиков
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # График 1: Конечный X
    ax1.plot(initial_heights, np.array(final_x)*1e3, label="$\Delta x$", color="blue")
    ax1.set_xlabel("$h$ $(м)$")
    ax1.set_ylabel("Конечный $x$ $(м)$")
    ax1.legend()
    ax1.set_xlim(0, max(initial_heights))
    ax1.set_ylim(0, None)
    ax1.text(0.005, 1.005, r"1e$-$3", transform=ax1.transAxes, fontsize=10, va='bottom', ha='left')
    # График 2: Конечный Y
    if len(initial_heights) == len(approximations):
        ax2.plot(initial_heights, approximations, label="Приближение", color="red")
    else:
        print("Размеры массивов не совпадают:", len(initial_heights), len(approximations))

    ax2.plot(initial_heights, final_y, label="$\Delta y$", color="green")
    ax2.plot(initial_heights, approximations, color="red")
    ax2.set_xlabel("$h$ (м)")
    ax2.set_ylabel("Конечный $y$ (м)")
    ax2.set_xlim(0, max(initial_heights))
    ax2.set_ylim(0, None)

    # Построение увеличенного участка
    zoom_ax = inset_axes(ax2, width="30%", height="30%",
                     bbox_to_anchor=(0.15, 0.1, 0.8, 0.8),
                     bbox_transform=ax2.transAxes,
                     loc='upper left')
    zoom_ax.clear()
    y_zoom_range = (final_y[10]-(final_y[10]-approximations[10]), final_y[10]+(final_y[10]-approximations[10]))  # Диапазон увеличения по x
    x_zoom_range = (initial_heights[10]-(final_y[10]-approximations[10])*10**3, initial_heights[10]+(final_y[10]-approximations[10])*10**3)
    zoom_ax.plot(initial_heights, final_y, color='green')
    zoom_ax.plot(initial_heights, approximations, color='red')
    zoom_ax.set_xlim(x_zoom_range)
    zoom_ax.set_ylim(y_zoom_range)

    x_ticks = np.linspace(x_zoom_range[0], x_zoom_range[1], 2)
    y_ticks = np.linspace(y_zoom_range[0], y_zoom_range[1], 2)

    zoom_ax.set_xticks(x_ticks)
    zoom_ax.set_yticks(y_ticks)

    zoom_ax.set_xticklabels([f"{tick:.2f}" for tick in x_ticks], fontsize=8)
    zoom_ax.set_yticklabels([f"{tick:.2e}" for tick in y_ticks], fontsize=8)

    zoom_ax.legend(fontsize=8)

    # График 4: Разность аппроксимации и Y
    ax3.plot(initial_heights, np.array(differences)*10**3, label="Разность (приближение - $\Delta y$)", color="purple")
    ax3.set_xlabel("$h$ $(м)$")
    ax3.text(0.09, 1.06, r"1e$-$3", transform=ax3.transAxes, fontsize=10, va='top', ha='right')
    ax3.set_ylabel("Разность $(м)$")
    ax3.legend()
    ax3.set_xlim(0, max(initial_heights))
    ax3.set_ylim(0, None)

    ax4.plot(initial_heights, differences_x, label="Разность (приближение - $\Delta x$)", color="purple")
    ax4.set_xlabel("$h$ $(м)$")
    ax4.set_ylabel("Разность $(м)$")
    ax4.set_xlim(0, max(initial_heights))
    ax4.set_ylim(0, None)
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
velocity_x_slider = Slider(ax_velocity_x, '$v_{0x}$ (м/с)', 0, 11000, valinit=0, valstep=1)
velocity_y_slider = Slider(ax_velocity_y, '$v_{0y}$ (м/с)', 0, 1000, valinit=0, valstep=0.1)
velocity_z_slider = Slider(ax_velocity_z, '$v_{oz}$ (м/с)', 0, 1000, valinit=0, valstep=0.1)
radius_slider = Slider(ax_radius, '$R_E$ (м)', 1e6, 1e7, valinit=R_0, valstep=1e5)
mass_slider = Slider(ax_mass, '$M_{E}$ (кг)', M_e, M_sun, valinit=M_e, valstep=(M_sun - M_e) / 100)


sliders_visible = True  # Ползунки отображаются по умолчанию




def toggle_sliders(event):
    global sliders_visible  # Объявляем использование глобальной переменной
    sliders_visible = not sliders_visible  # Переключаем видимость ползунков
    if sliders_visible:
        # Отображаем ползунки
        angle_slider.ax.set_visible(True)
        velocity_x_slider.ax.set_visible(True)
        velocity_y_slider.ax.set_visible(True)
        velocity_z_slider.ax.set_visible(True)
        radius_slider.ax.set_visible(True)
        mass_slider.ax.set_visible(True)
        toggle_button.label.set_text("⮝")  # Изменяем текст кнопки
        plt.subplots_adjust(left=0.25, right=0.95, bottom=0.35, top=0.95, hspace=0.4, wspace=0.4)
    else:
        angle_slider.ax.set_visible(False)
        velocity_x_slider.ax.set_visible(False)
        velocity_y_slider.ax.set_visible(False)
        velocity_z_slider.ax.set_visible(False)
        radius_slider.ax.set_visible(False)
        mass_slider.ax.set_visible(False)
        toggle_button.label.set_text("⮟")

        positions = [
            [0.1, 0.55, 0.35, 0.35],
            [0.55, 0.55, 0.35, 0.35],
            [0.1, 0.1, 0.35, 0.35], 
            [0.55, 0.1, 0.35, 0.35]  
        ]
        for i, ax in enumerate(fig.axes):
            if ax not in [angle_slider.ax, velocity_x_slider.ax, velocity_y_slider.ax,
                          velocity_z_slider.ax, radius_slider.ax, mass_slider.ax, ax_button]:
                ax.set_position(positions[i % len(positions)])

    plt.draw()

ax_button = plt.axes([0.01, 0.01, 0.05, 0.03])  # Уменьшаем размер кнопки и размещаем её в углу
toggle_button = Button(ax_button, "⮝", color='lightgray', hovercolor='gray')  # Менее заметная кнопка
toggle_button.on_clicked(toggle_sliders)



angle_slider.on_changed(update_plot)
velocity_x_slider.on_changed(update_plot)
velocity_y_slider.on_changed(update_plot)
velocity_z_slider.on_changed(update_plot)
radius_slider.on_changed(update_plot)
mass_slider.on_changed(update_plot)

# Инициализация графиков
update_plot(None)

plt.show()
