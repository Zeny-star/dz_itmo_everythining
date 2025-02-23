import numpy as np
import matplotlib.pyplot as plt

omega_0 = 1.0
f_0 = 1.0
beta = 0.1
omega = np.linspace(0, 2, 500)
delta = np.arctan2(2 * beta * omega, omega_0**2 - omega**2)

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(omega, delta, label="ФЧХ", color="orange")
plt.title("Фазо-частотная характеристика (ФЧХ)")
plt.xlabel("Частота ω")
plt.ylabel("Фаза δ(ω)")
plt.grid(True)
plt.legend()
plt.show()
