
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.widgets import Button

w = 7.2921e-5
R_0 = 6371e3
G = 6.67e-11
M_e = 5.97e24
M_sun = 1.989e30

def equations(t, state, R, M, pxi_0, omega):
    x, y, z, vx, vy, vz = state
    g_E0 = G * M / R**2
    alpha = omega**2 * R / g_E0 * np.sin(pxi_0) * np.cos(pxi_0)
    teta = pxi_0 - alpha
    w_x = -omega * np.sin(teta)
    w_z = omega * np.cos(teta)
    g_0 = g_E0 * (1 - omega**2 * R / g_E0 * np.sin(pxi_0)**2)
    g_x_x = -(g_E0 / R - w_z**2)
    g_y_y = -(g_E0 / R - omega**2)
    g_z_z = 2 * g_E0 / R + w_x**2
    g_x_z = 3 * alpha * g_E0 / R - w_z * w_x
    ax_1 = g_x_x * x + g_x_z * z + 2 * w_z * vy
    ay_1 = g_y_y * y - 2 * w_z * vx + 2 * w_x * vz
    az_1 = -g_0 + g_x_z * x + g_z_z * z - 2 * w_x * vy
    return [vx, vy, vz, ax_1, ay_1, az_1]

def equations_approx(t, state, R, M, pxi_0, omega):
    x, y, z, vx, vy, vz = state
    g_E0 = G * M / R**2
    alpha = omega**2 * R / g_E0 * np.sin(pxi_0) * np.cos(pxi_0)
    teta = pxi_0 - alpha
    w_x = -omega * np.sin(teta)
    w_z = omega * np.cos(teta)
    g_0 = g_E0 * (1 - omega**2 * R / g_E0 * np.sin(pxi_0)**2)
    g_x_z = 3 * alpha * g_E0 / R - w_z * w_x
    ax = g_x_z * z + 2 * w_z * vy
    ay = 2 * w_x * vz
    az = -g_0
    return [vx, vy, vz, ax, ay, az]

def calculate_final_coordinates_approx(initial_height, initial_angle, initial_velocity_x, initial_velocity_y, initial_velocity_z, planet_radius, planet_mass, omega):
    pxi_0 = np.radians(initial_angle)
    R = planet_radius
    M = planet_mass

    initial_state = [0, 0, initial_height, initial_velocity_x, initial_velocity_y, initial_velocity_z]

    def hit_ground(t, state):
        z = state[2]
        return z

    hit_ground.terminal = True
    hit_ground.direction = -1

    t_span = (0, 1e6)

    solution = solve_ivp(
        lambda t, state: equations_approx(t, state, R, M, pxi_0, omega),
        t_span,
        initial_state,
        method='DOP853',
        events=hit_ground
    )

    x_final, y_final = solution.y[0, -1], solution.y[1, -1]
    return x_final, y_final

def calculate_final_coordinates(initial_height, initial_angle, initial_velocity_x, initial_velocity_y, initial_velocity_z, planet_radius, planet_mass, omega):
    pxi_0 = np.radians(initial_angle)
    R = planet_radius
    M = planet_mass
    initial_state = [0, 0, initial_height, initial_velocity_x, initial_velocity_y, initial_velocity_z]
    g_E0 = G * M / R**2
    g_0 = g_E0 * (1 - omega**2 * R / g_E0 * np.sin(pxi_0)**2)
    t_span = (0, (2 * initial_state[2] / g_0)**0.5)
    t_eval = np.linspace(*t_span, 1000)
    solution = solve_ivp(equations, t_span, initial_state, t_eval=t_eval, method='DOP853', args=(R, M, pxi_0, omega))
    x_final, y_final = solution.y[0, -1], solution.y[1, -1]
    return x_final, y_final

def update_plot(val):
    global w
    w = omega_slider.val

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
            h, initial_angle, initial_velocity_x, initial_velocity_y, initial_velocity_z, planet_radius, planet_mass, w
        )
        x_app, y_app = calculate_final_coordinates_approx(
            h, initial_angle, initial_velocity_x, initial_velocity_y, initial_velocity_z, planet_radius, planet_mass, w
        )
        final_x.append(x)
        final_y.append(y)
        approximations.append(y_app)
        differences.append(y_app - y)
        differences_x.append(x_app - x)
    zoom_ax = inset_axes(ax2, width="30%", height="30%",
                     bbox_to_anchor=(0.15, 0.1, 0.8, 0.8),
                     bbox_transform=ax2.transAxes,
                     loc='upper left')

    zoom_ax.clear()
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    ax1.plot(initial_heights, np.array(final_x)*1e3, label="$\Delta x$", color="blue")
    ax1.set_xlabel("$h$ $(м)$")
    ax1.set_ylabel("Конечный $x$ $(м)$")
    ax1.legend()
    ax1.set_xlim(0, max(initial_heights))
    ax1.set_ylim(0, None)
    ax1.text(0.005, 1.005, r"1e$-$3", transform=ax1.transAxes, fontsize=10, va='bottom', ha='left')

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

    zoom_ax.plot(initial_heights, final_y, color='green')
    zoom_ax.plot(initial_heights, approximations, color='red')

    ax3.plot(initial_heights, np.array(differences)*10**3, label="Разность (приближение - $\Delta y$)", color="purple")
    ax3.set_xlabel("$h$ $(м)$")
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

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

plt.subplots_adjust(left=0.25, right=0.95, bottom=0.4, hspace=0.4, wspace=0.4)

ax_angle = plt.axes([0.25, 0.3, 0.65])

