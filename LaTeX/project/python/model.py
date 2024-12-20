import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

w = 7.2921e-5
gy_z=gx_y = 0.0
R_0 = 6371e3
G = 6.67e-11
M_e = 5.97e24
g_E0 = G*M_e/R_0**2

def g_calc(pxi_0):
    alpha = w**2*R_0/g_E0*np.sin(pxi_0)*np.cos(pxi_0)
    teta = pxi_0 - alpha
    w_x = -w*np.sin(teta)
    w_z = w*np.cos(teta)
    g_0 = g_E0*(1-w**2*R_0/g_E0*np.sin(pxi_0)**2)
    g_x_x = -(g_E0/R_0-w_z**2)
    g_y_y = -(g_E0/R_0-w**2)
    g_z_z = 2*g_E0/R_0+w_x**2
    g_x_z=3*alpha*g_E0/R_0-w_z*w_x
    return g_0, g_x_x, g_y_y, g_z_z, g_x_z


def equations(t, state):
    x, y, z, vx, vy, vz, pxi_0 = state
    alpha = w**2*R_0/g_E0*np.sin(pxi_0)*np.cos(pxi_0)
    teta = pxi_0 - alpha
    w_x = -w*np.sin(teta)
    w_z = w*np.cos(teta)
    g_0 = g_E0*(1-w**2*R_0/g_E0*np.sin(pxi_0)**2)
    g_x_x = -(g_E0/R_0-w_z**2)
    g_y_y = -(g_E0/R_0-w**2)
    g_z_z = 2*g_E0/R_0+w_x**2
    g_x_z=3*alpha*g_E0/R_0-w_z*w_x
    ax_1=g_x_x*x+g_x_z*z+2*w_z*vy
    ay_1=g_y_y*y-2*w_z*vx+2*w_x*vz
    az_1=-g_0+g_x_z*x+g_z_z*z-2*w_x*vy
    ax_2 = gx_y * y + g_x_z * z + 2 * w * vz
    ay_2 = gy_z * z + 2 * w * vx
    az_2 = -g_0
    return [vx, vy, vz, ax_1, ay_1, az_1]


initial_state = [0, 0, 100, 0, 0, 0, np.pi/4]
g_0 = g_E0*(1-w**2*R_0/g_E0*np.sin(initial_state[6])**2)
t_span = (0, (2*initial_state[2]/g_0)**(1/2))
t_eval = np.linspace(*t_span, 10000)

solution = solve_ivp(equations, t_span, initial_state, t_eval=t_eval, method='RK45')

x, y, z = solution.y[0], solution.y[1], solution.y[2]

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z, label="Траектория")
ax.set_xlabel("X (м)")
ax.set_ylabel("Y (м)")
ax.set_zlabel("Z (м)")
ax.set_title("Траектория тела в пространстве")
ax.legend()
plt.show()

