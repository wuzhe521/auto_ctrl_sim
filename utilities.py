import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

field_size  = {"x_min": -10, "x_max": 200, "y_min": -10, "y_max": 100}

map_radius = [50, 200, 500, 1000, 1500, 5000]
map_kappa = [1 / i for i in map_radius]
map_velocity = [20, 40, 60, 80, 100, 150]

MAX_Heading_Diff = 0.05
MAX_Dist_Diff = 10


def interp(x: list, y: list, x_data: float):
    if x_data < x[0]:
        return y[0]
    if x_data > x[-1]:
        return y[-1]
    for i in range(len(x)):
        if x[i] > x_data:
            return y[i - 1] + (y[i] - y[i - 1]) * (x_data - x[i - 1]) / (
                x[i] - x[i - 1]
            )
        return y[i]


def interpolate(x: list, y: list, x_data: float):
    return interp(x, y, x_data)


# check max velocity according to set radius
def max_velocity_vs_radius(radius: float):
    return interpolate(map_radius, map_velocity, radius)


def max_velocity_vs_kappa(kappa: float):
    return interpolate(map_kappa, map_velocity, kappa)


# check min radius according to set velocity
def min_radius_vs_velocity(velocity: float):
    return interpolate(map_velocity, map_radius, velocity)


def max_distance_diff():
    return MAX_Dist_Diff


def max_heading_diff():
    return MAX_Heading_Diff


def show_start_message(ax):
    t = ax.text(
        0.5,
        0.5,
        "Spark Group\n Sim Start",
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        color="white",
        size=36,
        fontdict=None,
    )
    t.set_path_effects(
        [path_effects.Stroke(linewidth=5, foreground="green"), path_effects.Normal()]
    )


def show_end_message(ax):
    t = ax.text(
        0.5,
        0.5,
        "Spark Group\n Sim End",
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        color="white",
        size=36,
        fontdict=None,
    )
    t.set_path_effects(
        [path_effects.Stroke(linewidth=5, foreground="red"), path_effects.Normal()]
    )


def show_time(ax, loc_x, loc_y, time):
    ax.text(
        loc_x,
        loc_y,
        s="t = " + str(time).split(".")[0] + "." + str(time).split(".")[1][:1],
        size = 10
    )
def show_info(ax, loc_x, loc_y, info):
    ax.text(
        loc_x,
        loc_y,
        s=info,
        size = 10
    )


# do some test
if __name__ == "__main__":
    x = max_velocity_vs_radius(500.0)
    print("max velocity: ", x, "km/h", "at radius: ", 500.0, "m")
