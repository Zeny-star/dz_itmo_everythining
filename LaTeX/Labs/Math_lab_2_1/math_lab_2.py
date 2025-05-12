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
    n = int(slider_n.val)
    if n < 1:
        n = 1

    x_eval_points = np.linspace(a, b, n + 1)
    dx = (b - a) / n

    for i, method in enumerate(methods):
        ax = axs[i]
        ax.clear()

        ax.plot(x_smooth, y_smooth, 'b-', linewidth=1.5, label='$f(x) = e^{3x}$', zorder=1)
        legend_handles = [plt.Line2D([0], [0], color='b', lw=1.5, label='$f(x) = e^{3x}$')]
        integral_sum = 0.0
        plot_title = ""
        color = method_colors[method]
        
        point_marker_size = 4 
        legend_point_marker_size = 4
        connecting_line_width = 1.0

        if method == 'left':
            xi = x_eval_points[:-1]
            heights = f(xi)
            integral_sum = np.sum(heights * dx)
            ax.bar(xi, heights, width=dx, alpha=0.5, align='edge', edgecolor='black', color=color, zorder=0.5)
            ax.plot(xi, heights, '-', color='black', linewidth=connecting_line_width, zorder=2)
            ax.plot(xi, heights, 'o', color='black', linestyle='None', markersize=point_marker_size, zorder=3)
            legend_handles.append(Patch(facecolor=color, alpha=0.5, edgecolor='black', label='Прямоугольники'))
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', linestyle='None', markersize=legend_point_marker_size, label='Точки оснащения'))

        elif method == 'right':
            xi = x_eval_points[1:]
            heights = f(xi)
            integral_sum = np.sum(heights * dx)
            ax.bar(x_eval_points[:-1], heights, width=dx, alpha=0.5, align='edge', edgecolor='black', color=color, zorder=0.5)
            ax.plot(xi, heights, '-', color='black', linewidth=connecting_line_width, zorder=2)
            ax.plot(xi, heights, 'o', color='black', linestyle='None', markersize=point_marker_size, zorder=3)
            legend_handles.append(Patch(facecolor=color, alpha=0.5, edgecolor='black', label='Прямоугольники'))
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', linestyle='None', markersize=legend_point_marker_size, label='Точки оснащения'))

        elif method == 'midpoint':
            xi = (x_eval_points[:-1] + x_eval_points[1:]) / 2
            heights = f(xi)
            integral_sum = np.sum(heights * dx)
            ax.bar(x_eval_points[:-1], heights, width=dx, alpha=0.5, align='edge', edgecolor='black', color=color, zorder=0.5)
            ax.plot(xi, heights, '-', color='black', linewidth=connecting_line_width, zorder=2)
            ax.plot(xi, heights, 'o', color='black', linestyle='None', markersize=point_marker_size, zorder=3)
            legend_handles.append(Patch(facecolor=color, alpha=0.5, edgecolor='black', label='Прямоугольники'))
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', linestyle='None', markersize=legend_point_marker_size, label='Точки оснащения'))

        elif method == 'trapezoid':
            y_endpoints = f(x_eval_points)
            integral_sum = np.sum((y_endpoints[:-1] + y_endpoints[1:]) / 2 * dx)
            for j in range(n):
                 verts = [(x_eval_points[j], 0), (x_eval_points[j], y_endpoints[j]), (x_eval_points[j+1], y_endpoints[j+1]), (x_eval_points[j+1], 0)]
                 poly = plt.Polygon(verts, facecolor=color, alpha=0.5, edgecolor='black', zorder=0.5)
                 ax.add_patch(poly)
            ax.plot(x_eval_points, y_endpoints, '-', color='black', linewidth=connecting_line_width, zorder=2)
            ax.plot(x_eval_points, y_endpoints, 'o', color='black', linestyle='None', markersize=point_marker_size, zorder=3)
            legend_handles.append(Patch(facecolor=color, alpha=0.5, edgecolor='black', label='Трапеции'))
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', linestyle='None', markersize=legend_point_marker_size, label='Точки оснащения'))


        plot_title = f'{method_titles[method]}, n={n}\nСумма = {integral_sum:.6f}'
        ax.set_title(plot_title, fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_xlim(a, b)
        ax.set_ylim(0, f(b) * 1.1)
        
        legend = ax.legend(handles=legend_handles, loc='upper left', frameon=True, fontsize=8.5, borderpad=0.5)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.6)
        legend.set_zorder(10)

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
