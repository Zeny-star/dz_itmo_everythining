import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Patch

def f(x):
    return np.exp(3 * x)

a = 0.0
b = 0.5
initial_n = 10
max_n = 200

methods = ['left', 'right', 'midpoint', 'trapezoid']
method_titles = {
    'left': 'Левая сумма',
    'right': 'Правая сумма',
    'midpoint': 'Средняя сумма',
    'trapezoid': 'Сумма трапеций'
}
method_colors = {
    'left': 'skyblue',
    'right': 'lightgreen',
    'midpoint': 'lightcoral',
    'trapezoid': 'orange'
}

x_smooth = np.linspace(a, b, 500)
y_smooth = f(x_smooth)

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
axs = axs.flatten()

plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, hspace=0.3, wspace=0.2)

def update(val):
    n = int(slider_n.val) #крутилка
    if n < 1:
        n = 1

    x = np.linspace(a, b, n + 1)
    dx = (b - a) / n

    for i, method in enumerate(methods):
        ax = axs[i]
        ax.clear()

        ax.plot(x_smooth, y_smooth, 'b-', linewidth=1.5, label='$f(x) = e^{3x}$')
        legend_handles = [plt.Line2D([0], [0], color='b', lw=1.5, label='$f(x) = e^{3x}$')]
        integral_sum = 0.0
        plot_title = ""
        color = method_colors[method]

        if method == 'left':
            xi = x[:-1]
            heights = f(xi)
            integral_sum = np.sum(heights * dx)
            ax.bar(xi, heights, width=dx, alpha=0.5, align='edge', edgecolor='black', color=color)
            ax.plot(xi, heights, 'black', markersize=3)
            legend_handles.append(Patch(facecolor=color, alpha=0.5, edgecolor='black', label='Прямоугольники'))
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=3, label='Точки оснащения'))

        elif method == 'right':
            xi = x[1:]
            heights = f(xi)
            integral_sum = np.sum(heights * dx)
            ax.bar(x[:-1], heights, width=dx, alpha=0.5, align='edge', edgecolor='black', color=color)
            ax.plot(xi, heights, 'black', markersize=3)
            legend_handles.append(Patch(facecolor=color, alpha=0.5, edgecolor='black', label='Прямоугольники'))
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=3, label='Точки оснащения'))

        elif method == 'midpoint':
            xi = (x[:-1] + x[1:]) / 2
            heights = f(xi)
            integral_sum = np.sum(heights * dx)
            ax.bar(x[:-1], heights, width=dx, alpha=0.5, align='edge', edgecolor='black', color=color)
            ax.plot(xi, heights, 'black', markersize=3)
            legend_handles.append(Patch(facecolor=color, alpha=0.5, edgecolor='black', label='Прямоугольники'))
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=3, label='Точки оснащения'))

        elif method == 'trapezoid':
            y_endpoints = f(x)
            integral_sum = np.sum((y_endpoints[:-1] + y_endpoints[1:]) / 2 * dx)
            for j in range(n):
                 verts = [(x[j], 0), (x[j], y_endpoints[j]), (x[j+1], y_endpoints[j+1]), (x[j+1], 0)]
                 poly = plt.Polygon(verts, facecolor=color, alpha=0.5, edgecolor='black')
                 ax.add_patch(poly)
            ax.plot(x, y_endpoints, 'black', markersize=3)
            legend_handles.append(Patch(facecolor=color, alpha=0.5, edgecolor='black', label='Трапеции'))
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=3, label='Точки оснащения'))


        plot_title = f'{method_titles[method]}, n={n}\nСумма = {integral_sum:.6f}'
        ax.set_title(plot_title, fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_xlim(a, b)
        ax.set_ylim(0, f(b) * 1.1)
        ax.legend(handles=legend_handles, fontsize=8)

    fig.canvas.draw_idle()

ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.03])

slider_n = Slider(
    ax=ax_slider,
    label='Число разбиений n',
    valmin=1,  
    valmax=max_n,
    valinit=initial_n,
    valstep=1        
)

slider_n.on_changed(update)

update(initial_n)

plt.show()

