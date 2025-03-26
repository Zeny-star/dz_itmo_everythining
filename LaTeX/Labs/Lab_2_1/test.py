import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

mmHg_to_Pa = 13.55 * 9.80665  
p0 = 1e5

data = {
    't1': {'temp': 21.5 + 273.15, 'p1': [35.6,14.6,2.9,-9.2,-27.7,-38.9,-48.1,-52.1],
                                  'p2': [22.6,0.3,-11,-20.2,-33.7,-31.5,-47.3,-52.1]},
    't2': {'temp': 31.2 + 273.15, 'p1': [18.2,1.9,-12.1,-23.3,-32,-39.6,-51.3,-55.1],
                        'p2': [12.9,-6.1,-18.6,-28.9,-37.4,-44.9,-50.2,-55.1]},
    't3': {'temp': 40.1 + 273.15, 'p1': [12.6,-4.2,-15.9,-27.9,-35.5,-42.5,-49.3,-53.2],
                        'p2': [12.5,-1.3,-15.2,-27,-35.4,-42.8,-48.7,-53.2]},
    't4': {'temp': 49.0 + 273.15, 'p1': [15.4,-1.7,-14.7,-25.4,-34.8,-41.4,-50.8,-55.3],
                        'p2': [11.7,-3.9,-18.5,-28.6,-37.2,-44.7,-50,-55.4]},
    't5': {'temp': 60.0 + 273.15, 'p1': [13.6,-4.5,-15.5,-27.6,-36.2,-42.4,-49,-53.3],
                        'p2': [13.6,-0.2,-16.7,-26.5,-35.5,-42.4,-48.1,-53.3]}
}

volumes = np.array([50, 60, 70, 80, 90, 100, 110, 120]) * 1e-6

table3 = {
    'volumes': [50e-6, 90e-6, 120e-6],
    '1/V': [],
    't_star': [],
    'a': [],
    'c': [],
    'delta_t_star': []
}


selected_volumes_idx = [0, 4, 7]  

plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, color in zip(selected_volumes_idx, colors):
    pressures = []
    temps = []
    
    for key in data:
        p_avg = (np.array(data[key]['p1'][idx]) + np.array(data[key]['p2'][idx]))/2
        p_total = p0 + p_avg * mmHg_to_Pa
        pressures.append(p_total)
        temps.append(data[key]['temp'])
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(temps, pressures)
    
    t_star = -intercept/slope
    
    table3['1/V'].append(1/volumes[idx])
    table3['t_star'].append(t_star)
    table3['a'].append(slope)
    table3['c'].append(intercept)
    
    plt.plot(temps, pressures, 'o', color=color, 
             label=f'V={volumes[idx]*1e6:.0f} мл')
    plt.plot(temps, slope*np.array(temps)+intercept, '--', color=color)

plt.xlabel('Температура, K')
plt.ylabel('Давление, Па')
plt.title('Зависимость давления от температуры для разных объемов')
plt.legend()
plt.grid(True)
plt.show()

x = np.array(table3['1/V'])
y = np.array(table3['t_star'])

slope_prime, intercept_prime, r_value, p_value, std_err = stats.linregress(x, y)

n = len(x)
x_mean = np.mean(x)
S_xx = np.sum((x - x_mean)**2)
delta_C_prime = std_err * np.sqrt(1/n + x_mean**2/S_xx)

table3['delta_t_star'] = [delta_C_prime]*3  

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'bo', label='Экспериментальные точки')
plt.plot(x, slope_prime*x + intercept_prime, 'r-', 
         label=f'Аппроксимация: y = {slope_prime:.2f}x + {intercept_prime:.2f}')

x_ext = np.linspace(0, x.max()*1.1, 100)
plt.plot(x_ext, slope_prime*x_ext + intercept_prime, 'r--')

plt.xlabel('1/V, 1/м³')
plt.ylabel('t*, K')
plt.title('Зависимость t* от обратного объема')
plt.legend()
plt.grid(True)
plt.show()

print("\nРезультаты расчета Таблицы 3:")
print(f"{'V, м³':<15} {'1/V, 1/м³':<15} {'t*, K':<15} {'Δ(t*), K':<15}")
for i in range(3):
    print(f"{table3['volumes'][i]:<15.2e} {table3['1/V'][i]:<15.2e} "
          f"{table3['t_star'][i]:<15.2f} {table3['delta_t_star'][i]:<15.2f}")

print(f"\nКоэффициенты регрессии t*(1/V):")
print(f"A' = {slope_prime:.2f} ± {std_err:.2f} K·м³")
print(f"C' = {intercept_prime:.2f} ± {delta_C_prime:.2f} K")
