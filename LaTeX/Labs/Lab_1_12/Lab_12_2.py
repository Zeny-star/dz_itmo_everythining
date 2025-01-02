import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


data = np.loadtxt('1.text')
t = data[:, 0]
U1 = data[:, 1]
U2 = data[:, 2]

g=9.81
D=77/1000
m=1018/1000
l=94/100
L=l+D/2
t = data[:, 0]
U1 = data[:, 1]
U2 = data[:, 2]
L_1=90/100
k=3.4

peaks_U1, _ = find_peaks(U1, distance=50, prominence=0.01, height=0.01)
peaks_U2, _ = find_peaks(U2, distance=50, prominence=0.01, height=0.01)

omega_0=np.sqrt(g/L)
omega_2_1=0
omega_2_2=0


if len(peaks_U1) > 1:
    distances_U1 = np.diff(t[peaks_U1])
    mean_distance_U1 = np.mean(distances_U1)
    omega_1=2*np.pi/mean_distance_U1
    print(f"Средний омега_1: {omega_1:.4f} секунд")

if len(peaks_U2) > 1:
    distances_U2 = np.diff(t[peaks_U2])
    mean_distance_U2 = np.mean(distances_U2)
    omega_2=2*np.pi/mean_distance_U2
    print(f"Средний omega_2: {omega_2:.4f} секунд")



t_smooth = np.linspace(t.min(), t.max(), 300)
spline_U1 = make_interp_spline(t, U1)
spline_U2 = make_interp_spline(t, U2)
U1_smooth = spline_U1(t_smooth)
U2_smooth = spline_U2(t_smooth)

kappa_1=np.sqrt((omega_2_1**2-omega_0**2)/2)
kappa_2=np.sqrt((omega_2_2**2-omega_0**2)/2)

k_1=kappa_1**2*m*L**2/L_1**2
k_2=kappa_2**2*m*L**2/L_1**2

omega_values_2_1 = [3.63, 3.57, 3.53, 3.47, 3.45]
omega_values_2_2 = [3.62, 3.59, 3.54, 3.51, 3.45]
L_1_values = [70/100, 75/100, 80/100, 85/100, 90/100]

plt.figure(figsize=(10, 5))
plt.plot(t_smooth, U1_smooth, label='U1(t)', color='red')
plt.plot(t_smooth, U2_smooth, label='U2(t)', color='orange')
plt.xlabel('Время t [с]')
plt.ylabel('Напряжение U [В]')
plt.title('Зависимость напряжений U1 и U2 от времени t')
plt.xlim(left=0)
plt.legend(loc='upper right')
plt.grid()
#plt.show()
# Визуализация
plt.plot(t, U1, label="U1(t)", color="red")
plt.plot(t, U2, label="U2(t)", color="orange")
plt.plot(t[peaks_U1], U1[peaks_U1], "x", label="Пики U1", color="blue")
plt.plot(t[peaks_U2], U2[peaks_U2], "x", label="Пики U2", color="green")
plt.title("Сигналы и пики")
plt.xlabel("Время [с]")
plt.ylabel("Напряжение U [В]")
plt.legend()
plt.grid()
plt.show()


def linear_func(x, k, b):
    return k * x + b

x = np.array([70/100, 75/100, 80/100, 85/100, 90/100])
x = x**2

y = omega_values_2_2
params, covariance = curve_fit(linear_func, x, y)
k_fit, b_fit = params

k_error = np.sqrt(covariance[0, 0])
b_error = np.sqrt(covariance[1, 1])

print(f"Коэффициент k: {k_fit:.4f} ± {k_error:.4f}")
print(f"Свободный член b: {b_fit:.4f} ± {b_error:.4f}")
print(f"Коэффициент жесткости k: {k_fit*(m*g**(1/2)*L**(3/2)):.4f} ± {k_error:.4f}")
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label="Данные ωn1(L^2)", color='blue')
plt.xlabel("L^2 [м]")
plt.ylabel("ωn1 [рад/с]")
plt.title("Зависимость ωn1 от L^2")
plt.legend()
plt.grid()
plt.plot(x, linear_func(x, k_fit, b_fit), '--', label=f"Fit: k={k_fit:.4f}, b={b_fit:.4f}", color='red')
plt.show()
