import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import find_peaks


data = np.loadtxt('3.1.text')
t = data[:, 0]
U1 = data[:, 1]
U2 = data[:, 2]

t_smooth = np.linspace(t.min(), t.max(), 300)
spline_U1 = make_interp_spline(t, U1)
spline_U2 = make_interp_spline(t, U2)
U1_smooth = spline_U1(t_smooth)
U2_smooth = spline_U2(t_smooth)



peaks_U1, _ = find_peaks(U1, distance=350, prominence=0.01, height=(-0.05, 0.05))
peaks_U2, _ = find_peaks(U2, distance=500, prominence=0.01, height=(-0.05, 0.05))

if len(peaks_U1) > 1:
    distances_U1 = np.diff(t[peaks_U1])
    mean_distance_U1 = np.mean(distances_U1)
    omega_1=2*np.pi/mean_distance_U1
    std_distance_U1 = np.std(distances_U1, ddof=1)

if len(peaks_U2) > 1:
    distances_U2 = np.diff(t[peaks_U2])
    mean_distance_U2 = np.mean(distances_U2)
    omega_2=2*np.pi/mean_distance_U2
    std_distance_U2 = np.std(distances_U2, ddof=1)

sigma_omega_1 = (2 * np.pi * std_distance_U1) / (mean_distance_U1**2)
print(f"Средний период 1: {mean_distance_U1:.4f} секунд")
print(f"Средняя частота 1: {omega_1:.4f} рад/с")
print(f"Доверительный интервал частоты 1 (95%): {omega_1:.4f} \u00B1 {1.96 * sigma_omega_1:.4f}")
sigma_omega_2 = (2 * np.pi * std_distance_U2) / (mean_distance_U2**2)
print(f"Средний период 2: {mean_distance_U2:.4f} секунд")
print(f"Средняя частота 2: {omega_2:.4f} рад/с")
print(f"Доверительный интервал частоты 2 (95%): {omega_2:.4f} \u00B1 {1.96 * sigma_omega_2:.4f}")

plt.plot(t, U2, label="U2(t)", color="red")
plt.plot(t[peaks_U2], U2[peaks_U2], "x", label="Пики U1", color="blue")
plt.title("Сигналы и пики")
plt.xlabel("Время [с]")
plt.ylabel("Напряжение U [В]")
plt.legend()
plt.grid()

plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t_smooth, U1_smooth, label='U2(t)', color='orange')
plt.xlabel('Время t [с]')
plt.ylabel('Напряжение U [В]')
plt.title('Зависимость напряжений U1 и U2 от времени t')
plt.xlim(left=0)
plt.legend(loc='upper right')
plt.grid()
