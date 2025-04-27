import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

colors = ['#5986e4', '#7d4dc8', '#b01bb3']

p_0 = 10**5
t_1 = 21.5+273.15
p_1_1 = np.array([35.6, 14.6, 2.9, -9.2, -27.7, -38.9, -48.1, -52.1])*13.55*9.819
p_1_2 = np.array([35.6, 0.3, -13, -25.2, -33.7, -41.5, -47.3, -52.1])*13.55*9.819

t_2 = 31.2+273.15
p_2_1 = np.array([18.2, 1.9, -12.1, -23.3, -32, -39.6, -51.3, -55.1])*13.55*9.819
p_2_2 = np.array([12.9, -6.1, -18.6, -28.9, -37.4, -44.9, -50.2, -55.1])*13.55*9.819

t_3 = 40.1+273.15
p_3_1 = np.array([12.6, -4.2, -15.9, -27.9, -35.5, -42.5, -49.3, -53.2])
p_3_2 = np.array([12.5, -1.3, -15.2, -27, -35.4, -42.8, -48.7, -53.2])


t_4 = 49+273.15
p_4_1 = np.array([15.4, -1.7, -14.7, -25.4, -34.8, -41.4, -50.8, -55.3])
p_4_2 = np.array([11.7, -3.9, -18.5, -28.6, -37.2, -44.7, -50, -55.4])

t_5 = 60+273.15
p_5_1 = np.array([13.6, -4.5, -15.5, -27.6, -36.2, -42.4, -49, -53.3])
p_5_2 = np.array([13.6, -0.2, -16.7, -26.5, -35.5, -42.4, -48.1, -53.3])

v = np.array([50, 60, 70, 80, 90, 100, 110, 120])
v = v * 1e-6

p_1 = p_0 + (p_2_1+p_2_2)/2
p_2 = p_0 + (p_2_1+p_2_2)/2
p_3 = p_0 + (p_3_1+p_3_2)/2
p_4 = p_0 + (p_4_1+p_4_2)/2
p_5 = p_0 + (p_5_1+p_5_2)/2
y_1 = 1/p_1
y_2 = 1/p_2
y_3 = 1/p_3
y_4 = 1/p_4
y_5 = 1/p_5

A = np.vstack([v, np.ones(len(v))]).T  
a_1, b_1 = np.linalg.lstsq(A, y_1, rcond=None)[0]  
a_2, b_2 = np.linalg.lstsq(A, y_2, rcond=None)[0]
a_3, b_3 = np.linalg.lstsq(A, y_3, rcond=None)[0]
a_4, b_4 = np.linalg.lstsq(A, y_4, rcond=None)[0]
a_5, b_5 = np.linalg.lstsq(A, y_5, rcond=None)[0]

plt.plot(y_4, v, 'bo', label='Экспериментальные точки', color = colors[0])
plt.plot(a_4*v + b_4,v, 'r-', label=f'МНК', color = colors[2])
plt.ylabel('Объем $v$ (м³)')
plt.xlabel('$1/p$ (1/Па)')
plt.legend()
plt.show()

t = np.array([t_1, t_2, t_3, t_4, t_5])
a = np.array([a_1, a_2, a_3, a_4, a_5])
slope, intercept, r_value, p_value, std_err = stats.linregress(t, a)

A = slope
C = intercept

t_abs_zero = -C / A
n = len(t)
x_mean = np.mean(t)
S_xx = np.sum((t - x_mean)**2)
delta_A = std_err * np.sqrt(n / S_xx)
delta_C = std_err * np.sqrt(1/n + x_mean**2/S_xx)

delta_t_abs_zero = t_abs_zero * np.sqrt((delta_A/A)**2 + (delta_C/C)**2)
print(t_abs_zero, delta_t_abs_zero)

plt.plot(t, a)
plt.ylabel('a')
plt.xlabel('t')
plt.show()

p_50 = np.array([p_1[0], p_2[0], p_3[0], p_4[0], p_5[0]])
p_90 = np.array([p_1[4], p_2[4], p_3[4], p_4[4], p_5[4]])
p_120 = np.array([p_1[7], p_2[7], p_3[7], p_4[7], p_5[7]])
p_50_kPa = p_50 / 1000
p_90_kPa = p_90 / 1000
p_120_kPa = p_120 / 1000
def linear_least_squares(x, y):
    """
    Performs linear least squares fit y = Ax + C.
    Uses formulas from the appendix (Eq 16-19).
    Returns A, C, delta_A, delta_C
    """
    n = len(x)
    if n < 3:
        print(f"Warning: Only {n} points provided. Cannot calculate uncertainty reliably.")
        A = np.polyfit(x, y, 1)[0]
        C = np.polyfit(x, y, 1)[1]
        return A, C, np.nan, np.nan
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    D = np.sum((x - x_bar)**2)
    if D == 0: return np.nan, np.nan, np.nan, np.nan
    A = np.sum((x - x_bar) * y) / D
    C = y_bar - A * x_bar
    residuals = y - (A * x + C)
    E = np.sum(residuals**2) / (n - 2)
    delta_A = np.sqrt(E / D)
    delta_C = np.sqrt(E * (1/n + x_bar**2 / D))
    return A, C, delta_A, delta_C

a_50, c_50_Pa, _, _ = linear_least_squares(t, p_50)
a_90, c_90_Pa, _, _ = linear_least_squares(t, p_90)
a_120, c_120_Pa, _, _ = linear_least_squares(t, p_120)

t_line = np.linspace(min(t) - 5, max(t) + 5, 100)

p_line_50_kPa = (a_50 * t_line + c_50_Pa) / 1000
p_line_90_kPa = (a_90 * t_line + c_90_Pa) / 1000
p_line_120_kPa = (a_120 * t_line + c_120_Pa) / 1000

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 0.8, 3)) # Example colormap

plt.scatter(t, p_50_kPa, label='V = 50 mL (Эксп. точки)', color=colors[0], s=50, zorder=3)
plt.scatter(t, p_90_kPa, label='V = 90 mL (Эксп. точки)', color=colors[1], s=50, zorder=3)
plt.scatter(t, p_120_kPa, label='V = 120 mL (Эксп. точки)', color=colors[2], s=50, zorder=3)

plt.plot(t_line, p_line_50_kPa, '--', label='V = 50 mL (МНК)', color=colors[0], linewidth=1.5, zorder=2)
plt.plot(t_line, p_line_90_kPa, '--', label='V = 90 mL (МНК)', color=colors[1], linewidth=1.5, zorder=2)
plt.plot(t_line, p_line_120_kPa, '--', label='V = 120 mL (МНК)', color=colors[2], linewidth=1.5, zorder=2)
plt.show()

t_vol = np.array([-c_50_Pa / a_50, -c_90_Pa / a_90, -c_120_Pa / a_120])
v_a = np.array([1/50, 1/90, 1/120])
D = 0
for i in v_a:
    D += (i - np.mean(v_a))**2

for i in range(len(v_a)):
    A += 1/D*(v_a[i] - np.mean(v_a))*t_vol[i]
C = np.mean(v) - A * np.mean(t_vol)
plt.plot(t_vol, v_a, 'bo', label='Экспериментальные точки', color = 'red')
plt.plot(-A*t_vol + C, v_a, 'r-', label=f'МНК', color = 'blue')
plt.show()




