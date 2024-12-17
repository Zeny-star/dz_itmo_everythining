from manimlib import *
from scipy.integrate import odeint, solve_ivp

config.background_color = "#a0a0a0"
def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def ode_solution_points(function, state0, time, dt=0.001):
    solution = solve_ivp(
        function,
        t_span=(0, time),
        y0=state0,
        t_eval=np.arange(0, time, dt)
    )
    return  solution.y.T

class OpeningManimExample(Scene):
    def construct(self):
        self.camera.background_color=BLACK
        text = Text("Lorenz attractor", font_size=50)

        axes = ThreeDAxes(
            x_range=(-50, 50, 5),
            y_range=(-50, 50, 5),
            z_range=(-50, 50, 5),
            height=16,
            width=16,
            depth=8,
        )
        axes.set_width(FRAME_WIDTH)
        axes.center()

        self.frame.reorient(43, 76, 1, IN, 10)
        self.add(axes)

        state0 = [10, 10, 10]
        points = ode_solution_points(lorenz_system, state0, 10)

        curve = VMobject().set_points_as_corners(axes.c2p(*points.T))
        curve.set_stroke(BLUE, 2)
        self.play(ShowCreation(curve, run_time=10, rate_func=linear))
        self.play(Write(text))
        self.wait(2)
