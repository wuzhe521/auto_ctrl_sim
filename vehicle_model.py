import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import sparse


class vehicle_status:
    def __init__(
        self,
        x: float,
        y: float,
        theta: float,
        kappa: float,
        velocity: float,
        acceleration: float,
    ):
        self.x = x
        self.y = y
        self.theta = theta
        self.kappa = kappa
        self.velocity = velocity
        self.acceleration = acceleration


class vehicle_model:
    def __init__(self, name, angle, kappa, velocity, acceleration, X, Y):
        self.name = name
        self.angle = angle
        self.kappa = kappa
        self.velocity = velocity
        self.acceleration = acceleration
        self.X = X
        self.Y = Y
        self.s = 0.0
        self.Width = 1.9
        self.Length = 5.0

    def kinematic_Update(
        self, kappa_rate: float = 0.0, acceleration: float = 0.0, dt: float = 0.2
    ):
        """
        generate kinematic motion using kinematic model
             input:
                kappa_rate: kappa rate
                dt: sample time
        """
        kappa_rate = max(min(kappa_rate, 0.05), -0.05)
        self.kappa = self.kappa + kappa_rate * dt
        delta_theta = self.kappa * self.velocity * dt
        self.Y = (
            self.Y
            + self.velocity * np.cos(self.angle) * delta_theta * dt
            + self.velocity * np.sin(self.angle) * dt
        )
        self.X = (
            self.X
            - self.velocity * np.sin(self.angle) * delta_theta * dt
            + self.velocity * np.cos(self.angle) * dt
        )
        self.angle = self.angle + delta_theta
        self.acceleration = acceleration
        self.velocity = self.velocity + acceleration * dt
        self.s = self.s + self.velocity * dt + 0.5 * acceleration * dt * dt

    def position(self):
        """
        get vehicle 2D position in global coordinate
            output:
                points: 4 corner points of vehicle in global coordinate
        """
        left_front_x = (
            self.X
            + self.Length * np.cos(self.angle)
            - self.Width / 2 * np.sin(self.angle)
        )
        left_front_y = (
            self.Y
            + self.Length * np.sin(self.angle)
            + self.Width / 2 * np.cos(self.angle)
        )
        left_front = (left_front_x, left_front_y)
        right_front_x = (
            self.X
            + self.Length * np.cos(self.angle)
            + self.Width / 2 * np.sin(self.angle)
        )
        right_front_y = (
            self.Y
            + self.Length * np.sin(self.angle)
            - self.Width / 2 * np.cos(self.angle)
        )
        right_front = (right_front_x, right_front_y)
        left_rear_x = self.X - self.Width / 2 * np.sin(self.angle)
        left_rear_y = self.Y + self.Width / 2 * np.cos(self.angle)
        left_rear = (left_rear_x, left_rear_y)
        right_rear_x = self.X + self.Width / 2 * np.sin(self.angle)
        right_rear_y = self.Y - self.Width / 2 * np.cos(self.angle)
        right_rear = (right_rear_x, right_rear_y)
        return [left_front, right_front, right_rear, left_rear]

    def get_vehicle_status(self):
        """
        get vehicle status
            output:
                vehicle_status: vehicle status
        """
        return vehicle_status(
            self.X, self.Y, self.angle, self.kappa, self.velocity, self.acceleration
        )
    def plot_vehicle(self, ax):
        loc = self.position()
        rect = patches.Polygon(
            loc,
            linewidth=2,
            edgecolor="blue",
            facecolor="lightblue",
            alpha=0.7,
        )
        ax.add_patch(rect)

if __name__ == "__main__":
    ego = vehicle_model(
        name="ego",
        angle=20.0,
        kappa=0.05,
        velocity=0.0,
        X=20.0,
        Y=20.0,
        acceleration=0.0,
    )

    fig, ax = plt.subplots()
    # set x-axis from -10 to 10
    ax.set_xlim(-10, 100)
    ax.set_ylim(-10, 100)
    ax.set_aspect("equal")
    plt.ion()  # 开启 交互模式
    for i in range(10):
        points = ego.position()
        print(points)
        rect = patches.Polygon(
            points, linewidth=2, edgecolor="blue", facecolor="lightblue", alpha=0.7
        )
        ax.add_patch(rect)
        ego.kinematic_Update(0, 10, 0.2)
        plt.pause(0.1)
    ego.kappa = 0.0
    ego.angle = 0.0
    ego.X = 20.0
    ego.Y = 20.0
    for i in range(10):
        points = ego.position()
        rect = patches.Polygon(
            points, linewidth=2, edgecolor="red", facecolor="red", alpha=0.7
        )
        ax.add_patch(rect)
        ego.kinematic_Update(0, 0.025)
        plt.pause(0.1)
    # 关闭交互模式
    plt.ioff()
    plt.show()
