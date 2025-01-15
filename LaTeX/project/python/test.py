
import numpy as np
import matplotlib.pyplot as plt

# Функция для аппроксимации

def approximate_function(x):
    return np.sin(x)  # Пример функции

def main():
    # Исходные данные
    x = np.linspace(0, 10, 100)
    y = approximate_function(x)

    # Данные аппроксимации
    x_approx = np.linspace(0, 10, 10)
    y_approx = approximate_function(x_approx)

    # Аппроксимированные значения на всей сетке
    y_interpolated = np.interp(x, x_approx, y_approx)

    # Разности между исходной функцией и аппроксимацией
    diff = y - y_interpolated

    # Построение графика
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))

    # Основной график функции и аппроксимации
    ax[0].plot(x, y, label='Original Function', color='blue')
    ax[0].plot(x, y_interpolated, label='Approximation', linestyle='--', color='orange')
    ax[0].scatter(x_approx, y_approx, label='Sample Points', color='red')
    ax[0].set_title('Function and Approximation')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend()
    ax[0].grid()

    # Добавление увеличенного участка на первом графике
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Участок для увеличения
    x_zoom_range = (2, 4)  # Диапазон увеличения по x
    zoom_ax = inset_axes(ax[0], width="40%", height="40%", loc='upper right')

    # Построение увеличенного участка
    zoom_ax.plot(x, y, color='blue')
    zoom_ax.plot(x, y_interpolated, linestyle='--', color='orange')
    zoom_ax.scatter(x_approx, y_approx, color='red')
    zoom_ax.set_xlim(x_zoom_range)
    zoom_ax.set_ylim(approximate_function(x_zoom_range[0]), approximate_function(x_zoom_range[1]))
    zoom_ax.set_title('Zoomed In')
    zoom_ax.grid()

    # Разности между функциями
    ax[1].plot(x, diff, label='Difference', color='green')
    ax[1].set_title('Difference Between Function and Approximation')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('Difference')
    ax[1].legend()
    ax[1].grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

