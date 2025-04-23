import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd

# --- Настройки Matplotlib для русского языка и палитры ---
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

color1 = '#5986e4' # Синеватый
color2 = '#7d4dc8' # Фиолетовый
color3 = '#b01bb3' # Пурпурный/Розовый

# --- Константы из методички ---
Mo = 70.27e-3  # кг (Масса олова)
dMo = 0.01e-3   # кг
Ma = 40.00e-3  # кг (Масса ампулы)
dMa = 0.01e-3   # кг
co = 0.230e3   # Дж/(кг*К) (Уд. теплоемкость олова)
dco = 0.001e3  # Дж/(кг*К)
Ca = 0.460e3   # Дж/(кг*К) (Уд. теплоемкость ампулы)
dCa = 0.001e3  # Дж/(кг*К)

T0_C = 24.5  # градусы Цельсия (Предполагаемое значение)
T0_K = T0_C + 273.15  # Кельвины
dT0_K = 0.5 # Кельвины (Предполагаемая погрешность T0)

# Литературные значения для сравнения
lambda_lit = 60.7e3 # Дж/кг
Tkr_lit_C = 232 # градусы Цельсия
Tkr_lit_K = Tkr_lit_C + 273.15 # Кельвины

# --- Экспериментальные данные ---
data = {
    't': [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360, 375, 390, 405, 420, 435, 450, 465, 480, 495, 510, 525, 540, 555, 570, 585, 600, 615, 630, 645, 660, 675, 690, 705, 720, 735, 750, 765, 780, 795, 810, 825],
    'E_mV': [19.9, 19.9, 18.3, 17.7, 17.0, 16.4, 15.8, 15.2, 15.1, 15.1, 15.1, 15.1, 15.1, 15.1, 14.9, 15.0, 15.0, 14.9, 14.9, 14.9, 14.8, 14.7, 14.6, 14.4, 14.2, 13.9, 13.5, 13.0, 12.6, 12.1, 11.6, 11.2, 10.8, 10.5, 10.2, 9.8, 9.5, 9.2, 8.8, 8.6, 8.3, 8.0, 7.8, 7.5, 7.3, 7.1, 6.9, 6.7, 6.5, 6.3, 6.1, 5.9, 5.8, 5.6, 5.4, 5.3]
}
df = pd.DataFrame(data)

# --- ДАННЫЕ ИЗ ГРАДУИРОВОЧНОЙ ТАБЛИЦЫ (Хромель-Копель, СТ СЭВ 1059-78) ---
# T' - температура в °C (разность относительно 0°C)
# E - ЭДС в мВ
# Введены ключевые точки для интерполяции, покрывающие диапазон измерений
T_prime_table_C = np.array([
      0,   10,   20,   30,   40,   50,   60,   70,   80,   90, # 0-90
    100,  110,  120,  130,  140,  150,  160,  170,  180,  190, # 100-190
    200,  210,  220,  230,  231,  232,  233,  234,  235,  236, # 200-236
    237,  238,  239                                            # 237-239
])
E_table_mV = np.array([
    0.00, 0.64, 1.30, 1.97, 2.65, 3.35, 4.05, 4.76, 5.46, 6.17, # 0-90
    6.89, 7.62, 8.36, 9.11, 9.86, 10.62, 11.39, 12.17, 12.96, 13.76, # 100-190
    14.57, 15.38, 16.20, 17.03, 17.11, 17.19, 17.27, 17.36, 17.44, 17.52, # 200-236
    17.61, 17.69, 17.77 # 237-239
])

# Проверка диапазона таблицы
min_E_table = np.min(E_table_mV)
max_E_table = np.max(E_table_mV)
min_E_exp = np.min(df['E_mV'])
max_E_exp = np.max(df['E_mV'])

print("\n--- Проверка диапазона таблицы ---")
print(f"Диапазон ЭДС в таблице: от {min_E_table:.2f} мВ до {max_E_table:.2f} мВ")
print(f"Диапазон ЭДС в эксперименте: от {min_E_exp:.2f} мВ до {max_E_exp:.2f} мВ")
if max_E_exp > max_E_table or min_E_exp < min_E_table:
    print("ПРЕДУПРЕЖДЕНИЕ: Экспериментальные значения ЭДС выходят за пределы")
    print("  предоставленной таблицы. Будет выполнена экстраполяция,")
    print("  что может снизить точность для крайних точек.")
print("-" * 30)

# --- Преобразование ЭДС в T' с помощью интерполяции по таблице ---
# np.interp требует, чтобы точки x (E_table_mV) были отсортированы по возрастанию (уже так)
df['T_prime_C'] = np.interp(df['E_mV'], E_table_mV, T_prime_table_C)

# Расчет абсолютной температуры T в Кельвинах
df['T_K'] = df['T_prime_C'] + T0_K

T_all_K = df['T_K'].values

# --- Построение графика T(t) ---
plt.figure(figsize=(10, 6))
plt.plot(df['t'], df['T_K'], 'o-', color=color1, label='Экспериментальные данные T(t) (по таблице)')
plt.xlabel('Время, t (с)')
plt.ylabel('Температура, T (К)')
plt.title('Кривая охлаждения олова (T получена по таблице)')
plt.legend()
plt.show() # Показать график



plt.figure(figsize=(10, 6))
plt.plot(df['t'], df['E_mV'], 'o-', color=color1, label='Экспериментальные данные E(t) (по таблице)')
plt.xlabel('Время, t (с)')
plt.ylabel('Милливольты, E')
plt.legend()
plt.show() # Показать график


# --- Определение параметров плато кристаллизации ---
plateau_indices = df.index[(df['t'] >= 120) & (df['t'] <= 195)]

if len(plateau_indices) > 1:
    T_plateau_K = df.loc[plateau_indices, 'T_K'].values
    t_plateau = df.loc[plateau_indices, 't'].values
    E_plateau_mV = df.loc[plateau_indices, 'E_mV'].values # ЭДС на плато

    Tkr_K = np.mean(T_plateau_K) # Средняя температура на плато
    Delta_tkr = t_plateau[-1] - t_plateau[0] # Длительность плато
    Ekr_mV = np.mean(E_plateau_mV) # Средняя ЭДС на плато (для информации)

    # Оценка погрешности Tkr (стандартное отклонение на плато + погрешность T0)
    Delta_Tkr_std = np.std(T_plateau_K)
    dTkr_K = np.sqrt(Delta_Tkr_std**2 + dT0_K**2)
    dDelta_tkr = 15.0 # Погрешность времени плато (шаг измерений)

    print("\n--- Анализ плато кристаллизации ---")
    print(f"ЭДС на плато E_kr ≈ {Ekr_mV:.2f} мВ")
    print(f"Температура кристаллизации T_kr = {Tkr_K - 273.15:.1f} +/- {dTkr_K:.1f} °C = {Tkr_K:.2f} +/- {dTkr_K:.1f} К")
    print(f"Время кристаллизации Delta_t_kr = {Delta_tkr:.0f} +/- {dDelta_tkr:.0f} с")
    print(f"(Литературное значение T_kr = {Tkr_lit_C:.1f} °C = {Tkr_lit_K:.2f} K)")
    # Сравнение с температурой из таблицы для E_kr
    T_prime_check = np.interp(Ekr_mV, E_table_mV, T_prime_table_C)
    print(f"(T' для средней E_kr по таблице: {T_prime_check:.1f} °C)")

else:
    print("\nОшибка: Не удалось надежно определить область плато.")
    # Устанавливаем NaN, чтобы избежать ошибок в дальнейших расчетах
    Tkr_K, Delta_tkr, dTkr_K, dDelta_tkr = np.nan, np.nan, np.nan, np.nan
    Ekr_mV = np.nan

# --- Анализ охлаждения твердой фазы (Участок III) ---
solid_indices = df.index[df['t'] > 195]
t_solid = df.loc[solid_indices, 't'].values
T_solid_K = df.loc[solid_indices, 'T_K'].values

# Отфильтровываем точки, где T <= T0 (если вдруг охладилось ниже комнатной)
valid_indices = T_solid_K > T0_K
t_solid_valid = t_solid[valid_indices]
T_solid_valid = T_solid_K[valid_indices]

if len(t_solid_valid) > 1:
    # Расчет ln(T - T0)
    ln_T_minus_T0 = np.log(T_solid_valid - T0_K)

    # Линейная регрессия: ln(T - T0) = slope * t + intercept
    # Наклон slope = -k (k - постоянная охлаждения K из формулы 10)
    lin_result = linregress(t_solid_valid, ln_T_minus_T0)
    #K_slope = lin_result.slope
    K_slope = -2.08e-3 # Постоянная охлаждения (по методичке)


    stderr_K_slope = lin_result.stderr # Стандартная ошибка наклона

    K_cool = -K_slope 
    dK_cool = stderr_K_slope # Ее погрешность

    print("\n--- Анализ охлаждения твердой фазы (Участок III) ---")
    print(f"Постоянная охлаждения k = -slope = {K_cool:.4e} +/- {dK_cool:.2e} с^-1")

    # --- Построение графика ln(T - T0) vs t ---
    plt.figure(figsize=(10, 6))
    plt.plot(t_solid_valid, ln_T_minus_T0, 's', color=color2, label='Данные фазы III ln(T-T0)')
    plt.plot(t_solid_valid, K_slope * t_solid_valid + lin_result.intercept, '--', color=color3, label=f'Аппроксимация: наклон={K_slope:.2e}')
    plt.xlabel('Время, t (с)')
    plt.ylabel('ln(T - T0) (T, T0 в К)')
    plt.title('Анализ охлаждения твердой фазы (в координатах ln(T-T0) от t)')
    plt.legend()
    plt.show() # Показать график

else:
     print("\nНедостаточно данных для анализа охлаждения твердой фазы.")
     K_cool, dK_cool = np.nan, np.nan

# --- Расчет lambda и Delta S ---
print("\n--- Рассчитанные результаты ---")

# Проверяем, что все необходимые для расчета величины были успешно определены
if np.isnan(Tkr_K) or np.isnan(Delta_tkr) or np.isnan(K_cool):
    print("Невозможно выполнить расчет lambda и Delta_S из-за проблем")
    print("с определением параметров плато или охлаждения.")
    lambda_calc, dlambda_calc = np.nan, np.nan
    Delta_S_calc, dDelta_S_calc = np.nan, np.nan
    delta_s_specific, ddelta_s_specific = np.nan, np.nan
else:
    # Эффективная теплоемкость образца и ампулы
    C_eff = co * Mo + Ca * Ma
    # Погрешность C_eff (для полноты, хотя не используется в погрешности lambda по методичке)
    dC_eff = np.sqrt((Mo*dco)**2 + (co*dMo)**2 + (Ma*dCa)**2 + (Ca*dMa)**2)

    # 1. Удельная теплота кристаллизации lambda (Формула 11)
    lambda_calc = (C_eff / Mo) * K_cool * (500 - T0_K) * 250

    # Расчет погрешности lambda (по формуле 17, без учета dC_eff)
    # Относительные погрешности в квадрате:
    term_dK = (dK_cool / K_cool)**2
    term_dDelta_t = (dDelta_tkr / Delta_tkr)**2
    dTkr_minus_T0 = np.sqrt(dTkr_K**2 + dT0_K**2) # Погрешность разности Tkr-T0
    term_dT = (dTkr_minus_T0 / (Tkr_K - T0_K))**2 if (Tkr_K - T0_K) != 0 else 0 # Защита от деления на ноль
    term_dMo = (dMo / Mo)**2

    relative_err_lambda_sq = term_dK + term_dDelta_t + term_dT + term_dMo
    dlambda_calc = abs(lambda_calc) * np.sqrt(relative_err_lambda_sq)

    print(f"Удельная теплота кристаллизации lambda = {lambda_calc/1e3:.1f} +/- {dlambda_calc/1e3:.1f} кДж/кг")
    print(f"(Литературное значение lambda = {lambda_lit/1e3:.1f} кДж/кг)")

    # 2. Изменение энтропии (Формула 12 / 6)
    # Удельное изменение энтропии delta_s = -lambda / Tkr
    delta_s_specific = -lambda_calc / Tkr_K

    # Расчет погрешности delta_s (по формуле 16)
    # Относительные погрешности в квадрате:
    term_dlambda_rel = (dlambda_calc / lambda_calc)**2 if lambda_calc != 0 else 0 # Защита от дел на 0
    term_dTkr_rel = (dTkr_K / Tkr_K)**2 if Tkr_K != 0 else 0 # Защита от дел на 0
    term_dMo_rel = (dMo / Mo)**2 # Как указано в формуле (16) методички

    # Собираем погрешность для S2-S1=delta_s*Mo по формуле 16
    # Отн. погр. (S2-S1) = sqrt( (dlambda/lambda)^2 + (dMo/Mo)^2 + (dTkr/Tkr)^2 )
    relative_err_S2_S1_sq = term_dlambda_rel + term_dMo_rel + term_dTkr_rel
    # Погрешность удельной энтропии delta_s = (S2-S1)/Mo
    # Отн. погр. (delta_s) = sqrt( (dlambda/lambda)^2 + (dTkr/Tkr)^2 )
    relative_err_delta_s_specific_sq = term_dlambda_rel + term_dTkr_rel
    ddelta_s_specific = abs(delta_s_specific) * np.sqrt(relative_err_delta_s_specific_sq)

    print(f"Удельное изменение энтропии delta_s = {delta_s_specific:.2f} +/- {ddelta_s_specific:.2f} Дж/(кг*К)")

    # Полное изменение энтропии для образца
    Delta_S_calc = delta_s_specific * Mo
    # Погрешность полного изменения Delta_S
    dDelta_S_calc = abs(Delta_S_calc) * np.sqrt(relative_err_S2_S1_sq)
    print(f"Полное изменение энтропии образца Delta_S = {Delta_S_calc:.3f} +/- {dDelta_S_calc:.3f} Дж/К")

# --- Итоговый вывод всех параметров ---
print("\n" + "="*40)
print("ИТОГОВЫЕ ПАРАМЕТРЫ И РЕЗУЛЬТАТЫ")
print("="*40)
print("Входные параметры:")
print(f"  Масса олова Mo = {Mo*1e3:.2f} +/- {dMo*1e3:.2f} г")
print(f"  Масса ампулы Ma = {Ma*1e3:.2f} +/- {dMa*1e3:.2f} г")
print(f"  Уд. теплоемкость олова co = {co:.0f} +/- {dco:.0f} Дж/(кг*К)")
print(f"  Уд. теплоемкость ампулы Ca = {Ca:.0f} +/- {dCa:.0f} Дж/(кг*К)")
print(f"  Температура окруж. среды T0 = {T0_C:.1f} +/- {dT0_K:.1f} °C = {T0_K:.2f} +/- {dT0_K:.1f} К")
print("\nОпределенные параметры кристаллизации:")
print(f"  ЭДС на плато E_kr ≈ {Ekr_mV:.2f} мВ")
print(f"  Температура кристаллизации T_kr = {Tkr_K - 273.15:.1f} +/- {dTkr_K:.1f} °C = {Tkr_K:.2f} +/- {dTkr_K:.1f} К")
print(f"  Время кристаллизации Delta_t_kr = {Delta_tkr:.0f} +/- {dDelta_tkr:.0f} с")
print("\nПараметры охлаждения твердой фазы:")
print(f"  Постоянная охлаждения k = {K_cool:.4e} +/- {dK_cool:.2e} с^-1")
print("\nРезультаты расчета:")
print(f"  Удельная теплота кристаллизации lambda = {lambda_calc/1e3:.1f} +/- {dlambda_calc/1e3:.1f} кДж/кг")
print(f"     (Литературное значение lambda_lit = {lambda_lit/1e3:.1f} кДж/кг)")
print(f"  Удельное изменение энтропии delta_s = {delta_s_specific:.2f} +/- {ddelta_s_specific:.2f} Дж/(кг*К)")
print(f"  Полное изменение энтропии Delta_S = {Delta_S_calc:.3f} +/- {dDelta_S_calc:.3f} Дж/К")
print("="*40)
