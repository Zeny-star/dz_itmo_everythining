import numpy as np
import matplotlib.pyplot as plt

def f_twice_diff(x):
    return x**3

def f_twice_diff_prime(x):
    return 3 * x**2

def f_once_diff(x):
    return x**2 * np.sin(1 / x) if x != 0 else 0

def f_once_diff_prime(x):
    return 2*x*np.sin(1/x)-np.cos(1/x) if x != 0 else 0

def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

h_values = np.logspace(-20, 0, 100, base=10)
x_values_twice_diff = [0.5, 1.0, 1.5]
x_values_once_diff = [0.1, 0.01, 0.001] 

errors_forward_twice = []
errors_backward_twice = []
errors_central_twice = []

for x in x_values_twice_diff:
    error_f = [np.abs(f_twice_diff_prime(x) - forward_difference(f_twice_diff, x, h)) for h in h_values]
    error_b = [np.abs(f_twice_diff_prime(x) - backward_difference(f_twice_diff, x, h)) for h in h_values]
    error_c = [np.abs(f_twice_diff_prime(x) - central_difference(f_twice_diff, x, h)) for h in h_values]
    errors_forward_twice.append(error_f)
    errors_backward_twice.append(error_b)
    errors_central_twice.append(error_c)

errors_forward_once = []
errors_backward_once = []
errors_central_once = []

for x in x_values_once_diff:
    error_f = [np.abs(f_once_diff_prime(x) - forward_difference(f_once_diff, x, h)) for h in h_values]
    error_b = [np.abs(f_once_diff_prime(x) - backward_difference(f_once_diff, x, h)) for h in h_values]
    error_c = [np.abs(f_once_diff_prime(x) - central_difference(f_once_diff, x, h)) for h in h_values]
    errors_forward_once.append(error_f)
    errors_backward_once.append(error_b)
    errors_central_once.append(error_c)

def plot_errors(h_values, errors, x_values, title, ylabel="Ошибка"):
    plt.figure(figsize=(10, 6))
    for i, x in enumerate(x_values):
        plt.loglog(h_values, errors[i], label=f"x = {x:.2f}")
    plt.xlabel("Шаг (h)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.show()

# Plot for twice differentiable function
plot_errors(h_values, errors_forward_twice, x_values_twice_diff, "Разность вперед (два раза дифференцируемая)")
plot_errors(h_values, errors_backward_twice, x_values_twice_diff, "Разность назад (два раза дифференцируемая)")
plot_errors(h_values, errors_central_twice, x_values_twice_diff, "Центральная разность (два раза дифференцируемая)")

# Plot for once differentiable function
plot_errors(h_values, errors_forward_once, x_values_once_diff, "Разность вперед (один раз дифференцируемая)")
plot_errors(h_values, errors_backward_once, x_values_once_diff, "Разность назад (один раз дифференцируемая)")
plot_errors(h_values, errors_central_once, x_values_once_diff, "Центральная разность (один раз дифференцируемая)")
