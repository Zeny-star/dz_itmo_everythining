import numpy as np
import matplotlib.pyplot as plt
from scipy import stats # Keep for potential alternative checks, but use custom for uncertainty

# --- Constants ---
RHO_HG = 13.55 * 1000 # Density of Mercury (kg/m^3) - using the factor given
G = 9.819             # Acceleration due to gravity (m/s^2)
CONVERSION_FACTOR = RHO_HG * G * 1e-3 # Converts mm Hg to Pa (13.55 * 9.819 / 1000 * 1000)

# --- Input Data ---
# !!! YOU MUST REPLACE THIS WITH YOUR MEASURED VALUE !!!
p0_mmHg = 755.0  # Example: Atmospheric pressure in mm Hg
p0_Pa = p0_mmHg * CONVERSION_FACTOR

# Temperatures (°C and K)
t_c = np.array([21.5, 31.2, 40.1, 49.0, 60.0])
t_k = t_c + 273.15 # Keep Kelvin for reference if needed

# Cylinder Volumes (mL and m^3)
v_ml = np.array([50, 60, 70, 80, 90, 100, 110, 120])
v_m3 = v_ml * 1e-6

# Pressure Differences (Delta p in mm Hg)
# CORRECTION: Apply conversion factor to ALL readings
p_delta_raw = {
    't1': (np.array([35.6, 14.6, 2.9, -9.2, -27.7, -38.9, -48.1, -52.1]),
           np.array([35.6, 0.3, -13, -25.2, -33.7, -41.5, -47.3, -52.1])),
    't2': (np.array([18.2, 1.9, -12.1, -23.3, -32, -39.6, -51.3, -55.1]),
           np.array([12.9, -6.1, -18.6, -28.9, -37.4, -44.9, -50.2, -55.1])),
    't3': (np.array([12.6, -4.2, -15.9, -27.9, -35.5, -42.5, -49.3, -53.2]),
           np.array([12.5, -1.3, -15.2, -27, -35.4, -42.8, -48.7, -53.2])),
    't4': (np.array([15.4, -1.7, -14.7, -25.4, -34.8, -41.4, -50.8, -55.3]),
           np.array([11.7, -3.9, -18.5, -28.6, -37.2, -44.7, -50, -55.4])),
    't5': (np.array([13.6, -4.5, -15.5, -27.6, -36.2, -42.4, -49, -53.3]),
           np.array([13.6, -0.2, -16.7, -26.5, -35.5, -42.4, -48.1, -53.3]))
}

# Calculate Absolute Pressure (Pa) and Inverse Pressure (1/Pa)
# Store in 2D arrays: p_abs_Pa[temp_index, vol_index]
p_abs_Pa = np.zeros((len(t_c), len(v_ml)))
inv_p = np.zeros((len(t_c), len(v_ml)))

print("--- Calculated Absolute Pressures (kPa) ---")
for i, key in enumerate(['t1', 't2', 't3', 't4', 't5']):
    p1_mmhg, p2_mmhg = p_delta_raw[key]
    avg_delta_p_mmhg = (p1_mmhg + p2_mmhg) / 2
    avg_delta_p_Pa = avg_delta_p_mmhg * CONVERSION_FACTOR
    p_abs_Pa[i, :] = p0_Pa + avg_delta_p_Pa
    inv_p[i, :] = 1.0 / p_abs_Pa[i, :]
    print(f"T = {t_c[i]:.1f}°C: {np.round(p_abs_Pa[i,:]/1000, 1)}")
print("-" * 40)

# --- Least Squares Fitting Function (using Appendix Formulas Eq 16-19) ---
def linear_least_squares(x, y):
    """
    Performs linear least squares fit y = Ax + C.
    Uses formulas from the appendix (Eq 16-19).
    Returns A, C, delta_A, delta_C
    """
    n = len(x)
    if n < 3: # Need at least 3 points for N-2 degrees of freedom
        print(f"Warning: Only {n} points provided. Cannot calculate uncertainty reliably.")
        # Calculate fit anyway, return NaN for uncertainties
        A = np.polyfit(x, y, 1)[0]
        C = np.polyfit(x, y, 1)[1]
        return A, C, np.nan, np.nan

    x_bar = np.mean(x)
    y_bar = np.mean(y)

    D = np.sum((x - x_bar)**2)
    if D == 0:
        print("Warning: Cannot fit, all x values are the same.")
        return np.nan, np.nan, np.nan, np.nan

    A = np.sum((x - x_bar) * y) / D
    C = y_bar - A * x_bar

    # Calculate uncertainty (Eq 19, 18)
    residuals = y - (A * x + C)
    # Variance estimate E (sometimes called s^2_y or MSE)
    E = np.sum(residuals**2) / (n - 2)

    delta_A = np.sqrt(E / D)
    delta_C = np.sqrt(E * (1/n + x_bar**2 / D))

    return A, C, delta_A, delta_C

# --- Method 1: V_c vs 1/p (Corrected Approach) ---
print("\n--- Method 1: V_c vs 1/p Analysis ---")
K_values = []
delta_K_values = []
colors_m1 = plt.cm.viridis(np.linspace(0, 1, len(t_c)))
plt.figure(figsize=(10, 6))

for i in range(len(t_c)):
    x_data = inv_p[i, :] # 1/p
    y_data = v_m3        # V in m^3

    # Fit V = K*(1/p) + C_fit
    K, C_fit, delta_K, delta_C_fit = linear_least_squares(x_data, y_data)

    if not np.isnan(K):
        K_values.append(K)
        delta_K_values.append(delta_K)
        print(f"T = {t_c[i]:.1f}°C: K = {K:.3e} +/- {delta_K:.1e} J") # K = V*p -> units of Energy (Joule)

        # Plot data points (convert V to mL for readability)
        plt.scatter(x_data * 1e5, y_data * 1e6, label=f'T = {t_c[i]:.1f}°C', color=colors_m1[i], s=30)
        # Plot fit line
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = K * x_fit + C_fit
        plt.plot(x_fit * 1e5, y_fit * 1e6, '--', color=colors_m1[i], linewidth=1)
    else:
        K_values.append(np.nan)
        delta_K_values.append(np.nan)
        print(f"T = {t_c[i]:.1f}°C: Could not fit.")

plt.xlabel('Inverse Pressure (1/p) [10$^{-5}$ Pa$^{-1}$]')
plt.ylabel('Volume (V$_c$) [mL]')
plt.title('Method 1: V vs. 1/p at Constant Temperatures')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Fit K vs t (Method 1 continuation) ---
K_values = np.array(K_values)
delta_K_values = np.array(delta_K_values)
valid_indices_m1 = ~np.isnan(K_values) # Ensure we only use valid fits

if np.sum(valid_indices_m1) >= 3: # Need points to fit
    t_c_valid_m1 = t_c[valid_indices_m1]
    K_valid = K_values[valid_indices_m1]
    # delta_K_valid = delta_K_values[valid_indices_m1] # Not used in simple fit, but good practice for weighted fit

    # Fit K = A_K*t + C_K (where t is in Celsius)
    A_Kfit, C_Kfit, delta_A_Kfit, delta_C_Kfit = linear_least_squares(t_c_valid_m1, K_valid)

    print("\nFitting K(t) = A_K*t + C_K (t in Celsius):")
    print(f"  Slope A_K     = {A_Kfit:.3e} +/- {delta_A_Kfit:.1e} J/°C")
    print(f"  Intercept C_K = {C_Kfit:.3e} +/- {delta_C_Kfit:.1e} J")

    # Calculate t* where K(t*) = 0 => t* = -C_K / A_K
    t_star_1 = -C_Kfit / A_Kfit

    # Calculate uncertainty using Eq. 13 (approximate formula)
    # Delta_t* = |t*| * sqrt((Delta_A/A)^2 + (Delta_C/C)^2)
    if A_Kfit != 0 and C_Kfit != 0 and not np.isnan(delta_A_Kfit) and not np.isnan(delta_C_Kfit):
        # Using the robust formula (derived from error propagation for ratio)
        delta_t_star_1 = (1 / abs(A_Kfit)) * np.sqrt(delta_C_Kfit**2 + (t_star_1**2 * delta_A_Kfit**2))
        # Alternatively, using the manual's approx formula (Eq 13):
        # delta_t_star_1 = abs(t_star_1) * np.sqrt((delta_A_Kfit / A_Kfit)**2 + (delta_C_Kfit / C_Kfit)**2)
    else:
        delta_t_star_1 = np.nan

    print(f"\nResult Method 1:")
    print(f"  Absolute Zero t*_1 = {t_star_1:.2f} +/- {delta_t_star_1:.2f} °C")

    # Plot K vs t
    plt.figure(figsize=(8, 6))
    # Note: Plotting K with error bars, though simple LSQ doesn't use them
    plt.errorbar(t_c_valid_m1, K_valid, yerr=delta_K_values[valid_indices_m1], fmt='o', label='Data (K vs t)', capsize=5, markersize=6)
    # Extend fit line for extrapolation
    t_fit_range_m1 = np.linspace(min(t_c_valid_m1) - abs(t_star_1)*0.1 - 10, max(t_c_valid_m1)+10, 200)
    # Add extrapolation range if t_star_1 is outside the initial range
    t_fit_range_m1 = np.insert(t_fit_range_m1, 0, t_star_1 - 10)
    t_fit_range_m1 = np.sort(t_fit_range_m1)

    K_fit_line = A_Kfit * t_fit_range_m1 + C_Kfit
    plt.plot(t_fit_range_m1, K_fit_line, '--', label=f'Fit: K = {A_Kfit:.2e}*t + {C_Kfit:.2e}')
    plt.scatter([t_star_1], [0], color='red', s=100, zorder=5, label=f'Extrapolated t* = {t_star_1:.2f}°C')
    plt.axhline(0, color='grey', linestyle=':', linewidth=0.7)
    plt.axvline(0, color='grey', linestyle=':', linewidth=0.7)
    plt.xlabel('Temperature (t) [°C]')
    plt.ylabel('Slope K [J]') # K = V*p
    plt.title('Method 1: Slope K vs. Temperature t')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(bottom=min(0, min(K_valid)*1.1) if not np.isnan(min(K_valid)) else -0.1,
             top=max(K_valid)*1.1 if not np.isnan(max(K_valid)) else 0.1) # Adjust ylim
    plt.xlim(left=min(min(t_fit_range_m1), t_star_1-10), right=max(t_fit_range_m1)) # Adjust xlim for extrapolation
    plt.tight_layout()
    plt.show()

else:
    print("\nNot enough valid K values to perform K vs t fit for Method 1.")
    t_star_1, delta_t_star_1 = np.nan, np.nan

# --- Method 2: p vs t Analysis ---
print("\n--- Method 2: p vs t Analysis ---")
# Volumes specified in manual: 50, 90, 120 mL
# Find indices corresponding to these volumes in v_ml
selected_vol_ml = [50, 90, 120]
selected_vol_indices = [np.where(v_ml == v)[0][0] for v in selected_vol_ml if v in v_ml]

if not selected_vol_indices:
     print("Error: Could not find specified volumes (50, 90, 120 mL) in the data.")
else:
    selected_v_m3 = v_m3[selected_vol_indices]
    inv_selected_v_m3 = 1.0 / selected_v_m3 # Units: 1/m^3

    t_star_tilde = [] # Stores the t* intercept for each volume
    # delta_t_star_tilde = [] # Uncertainty of each intercept (optional to calculate here)

    colors_m2_pvst = plt.cm.plasma(np.linspace(0, 1, len(selected_vol_indices)))
    plt.figure(figsize=(10, 6))

    print("\nStep 7: Fit p(t) = a*t + c for constant volumes and find t*_tilde = -c/a")
    for idx, vol_idx in enumerate(selected_vol_indices):
        volume_ml = v_ml[vol_idx]
        pressures_Pa_constV = p_abs_Pa[:, vol_idx] # Get pressures for this volume across all temps
        temperatures_C_constV = t_c

        # Fit p = a*t + c (where t is in Celsius)
        # Use the custom function to potentially get uncertainties if needed later
        a_pfit, c_pfit, delta_a_pfit, delta_c_pfit = linear_least_squares(temperatures_C_constV, pressures_Pa_constV)

        if not np.isnan(a_pfit):
            print(f"  Volume = {volume_ml} mL:")
            # print(f"    Fit: p = ({a_pfit:.1f} +/- {delta_a_pfit:.1f})*t + ({c_pfit:.0f} +/- {delta_c_pfit:.0f})") # Verbose output

            # Calculate t*_tilde where p(t*) = 0 => t*_tilde = -c/a
            t_tilde = -c_pfit / a_pfit
            t_star_tilde.append(t_tilde)
            print(f"    Extrapolated t*_tilde = {t_tilde:.2f} °C")

            # Plot data points (Pressure in kPa)
            plt.scatter(temperatures_C_constV, pressures_Pa_constV / 1000,
                        label=f'V = {volume_ml} mL', color=colors_m2_pvst[idx], s=30)
            # Plot fit line and extrapolation
            t_fit_range_p = np.linspace(t_tilde - 20 , max(temperatures_C_constV) + 10, 100)
            # Ensure the fit range covers the calculated t_tilde for good visualization
            t_fit_range_p = np.insert(t_fit_range_p, 0, t_tilde - 10)
            t_fit_range_p = np.sort(t_fit_range_p)
            p_fit_line = a_pfit * t_fit_range_p + c_pfit
            plt.plot(t_fit_range_p, p_fit_line / 1000, '--', color=colors_m2_pvst[idx], linewidth=1)
            # Mark intercept t_tilde
            plt.scatter([t_tilde], [0], marker='x', color=colors_m2_pvst[idx], s=80, zorder=5)

        else:
            t_star_tilde.append(np.nan)
            print(f"  Volume = {volume_ml} mL: Could not fit p vs t.")

    plt.xlabel('Temperature (t) [°C]')
    plt.ylabel('Pressure (p) [kPa]')
    plt.title('Method 2: p vs. t at Constant Volumes (Extrapolating to find t*$_\\tilde$)')
    plt.axhline(0, color='grey', linestyle=':', linewidth=0.7)
    plt.axvline(0, color='grey', linestyle=':', linewidth=0.7)
    # Adjust limits for better view of extrapolation
    all_t_tilde_valid = [t for t in t_star_tilde if not np.isnan(t)]
    if all_t_tilde_valid:
        plt.xlim(left=min(min(temperatures_C_constV), min(all_t_tilde_valid))-20)
    plt.ylim(bottom=min(0, min(p_abs_Pa[:, selected_vol_indices].flatten()/1000)*1.1))
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # --- Step 8 & 9: Fit t*_tilde vs 1/V_c ---
    t_star_tilde = np.array(t_star_tilde)
    valid_indices_m2 = ~np.isnan(t_star_tilde) # Indices where t_tilde could be calculated

    print("\nStep 8 & 9: Fit t*_tilde vs 1/V_c and find t* = C'")
    if np.sum(valid_indices_m2) >= 3: # Need points to fit
        inv_v_valid_m2 = inv_selected_v_m3[valid_indices_m2] # x-data (1/m^3)
        t_tilde_valid = t_star_tilde[valid_indices_m2]       # y-data (°C)

        # Fit t*_tilde = A'*(1/V) + C' using formulas (16)-(19)
        # Requires the custom function to get delta_C' directly
        A_prime, C_prime, delta_A_prime, delta_C_prime = linear_least_squares(inv_v_valid_m2, t_tilde_valid)

        print(f"  Fit: t*_tilde = A' * (1/V) + C'")
        print(f"    Slope A'     = {A_prime:.3e} +/- {delta_A_prime:.1e} °C m^3")
        print(f"    Intercept C' = {C_prime:.2f} +/- {delta_C_prime:.2f} °C")

        # The intercept C' is the estimate for t* from Method 2
        t_star_2 = C_prime
        # The uncertainty delta_C' is the uncertainty delta_t* (Eq 18)
        delta_t_star_2 = delta_C_prime

        print(f"\nResult Method 2:")
        print(f"  Absolute Zero t*_2 = {t_star_2:.2f} +/- {delta_t_star_2:.2f} °C")

        # Plot t*_tilde vs 1/V
        plt.figure(figsize=(8, 6))
        # Convert 1/V for plotting readability (1 / mL)
        inv_v_plot = inv_v_valid_m2 / 1e6 # Units: 1/mL
        # Plot points (no error bars calculated for t_tilde easily here)
        plt.plot(inv_v_plot, t_tilde_valid, 'o', label='Data (t*$_\\tilde$ vs 1/V)', markersize=6)

        # Plot fit line and extrapolation to 1/V = 0
        # Create a range for 1/V including 0
        inv_v_fit_range_plot = np.linspace(-0.002, max(inv_v_plot) * 1.1, 100) # Range in 1/mL
        inv_v_fit_range_calc = inv_v_fit_range_plot * 1e6 # Convert back to 1/m^3 for calc
        t_tilde_fit_line = A_prime * inv_v_fit_range_calc + C_prime
        plt.plot(inv_v_fit_range_plot, t_tilde_fit_line, '--', label=f'Fit: t*$_\\tilde$ = {A_prime:.2e}*(1/V) + {C_prime:.2f}')

        # Mark intercept C' = t*_2
        plt.scatter([0], [t_star_2], color='red', s=100, zorder=5, label=f'Extrapolated t* = {t_star_2:.2f}°C')

        plt.axhline(0, color='grey', linestyle=':', linewidth=0.7)
        plt.axvline(0, color='grey', linestyle=':', linewidth=0.7)
        plt.xlabel('Inverse Volume (1/V$_c$) [mL$^{-1}$]')
        plt.ylabel('Extrapolated Temperature (t*$_\\tilde$) [°C]')
        plt.title('Method 2: Extrapolated Temp. t*$_\\tilde$ vs. Inverse Volume')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        # Adjust limits
        plt.xlim(left=min(inv_v_fit_range_plot), right=max(inv_v_fit_range_plot))
        plt.ylim(bottom=min(min(t_tilde_valid), t_star_2) - 10, top=max(t_tilde_valid)*1.1)
        plt.tight_layout()
        plt.show()

    else:
        print("\nNot enough valid t*_tilde values to perform t*_tilde vs 1/V fit for Method 2.")
        t_star_2, delta_t_star_2 = np.nan, np.nan


# --- Final Summary ---
print("\n" + "="*40)
print("          FINAL RESULTS SUMMARY")
print("="*40)
print(f"Method 1 (K vs t extrapolation):      t*_1 = {t_star_1:.2f} +/- {delta_t_star_1:.2f} °C")
print(f"Method 2 (t_tilde vs 1/V extrap.):    t*_2 = {t_star_2:.2f} +/- {delta_t_star_2:.2f} °C")
print("-"*40)
accepted_t_star = -273.15
print(f"Accepted value:                         t*   = {accepted_t_star:.2f} °C")
print("-"*40)

# Compare results
if not np.isnan(t_star_1):
    diff1 = abs(t_star_1 - accepted_t_star)
    print(f"Difference from accepted (Method 1): {diff1:.2f} °C")
    if not np.isnan(delta_t_star_1):
        print(f"  Compatibility Check (within ~3 sigma): {'Yes' if diff1 <= 3*delta_t_star_1 else 'No'}")

if not np.isnan(t_star_2):
    diff2 = abs(t_star_2 - accepted_t_star)
    print(f"Difference from accepted (Method 2): {diff2:.2f} °C")
    if not np.isnan(delta_t_star_2):
        print(f"  Compatibility Check (within ~3 sigma): {'Yes' if diff2 <= 3*delta_t_star_2 else 'No'}")
print("="*40)
