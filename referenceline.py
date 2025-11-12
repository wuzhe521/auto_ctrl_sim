import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from math import *
from utilities import *
from vehicle_model import vehicle_status



Safe_distance = 0.5


class Point:
    def __init__(
        self,
        x: float,
        y: float,
        dy: float,
        ddy: float,
        kappa: float,
        dkappa: float,
        s: float,
        angle: float,
    ):
        self.x = x
        self.y = y
        self.dy = dy
        self.ddy = ddy
        self.kappa = kappa
        self.dkappa = dkappa
        self.s = s
        self.angle = angle


class reference_line:
    def __init__(self, a0: float, a1: float, a2: float):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.points = self.get_ref_points(500)

    def get_point(self, dist: float):
        """
        根据给定距离[X - direction]，返回参考线点。
            Args:
                dist (float): X - direction distance.
            Returns:
                x, y, dy, ddy, kappa, dkappa, angle
        """
        x = dist
        y = self.a0 + self.a1 * dist + self.a2 * dist**2
        dy = self.a1 + 2 * self.a2 * dist
        ddy = 2 * self.a2
        kappa = dy / (1 + dy**2) ** (3 / 2)
        dkappa = (ddy * dy - dy**3) / (1 + dy**2) ** (5 / 2)
        angle = atan(dy)
        return x, y, dy, ddy, kappa, dkappa, angle

    def get_ref_points(self, pre_view_d: float):
        """
        根据预瞄的距离，生成参考线。

            Args:
                preview_d (float): Global 坐标下, X方向上的预瞄的距离。

            Returns:
                points (list) : 离散后的参考线点集合。
        """
        points = []
        x_scat = np.linspace(0, pre_view_d, 500)
        for i in range(len(x_scat)):
            if i == 0:
                x, y, dy, ddy, kappa, dkappa, angle = self.get_point(x_scat[i])
                points.append(Point(x, y, dy, ddy, kappa, dkappa, 0.0, angle))
            else:
                x, y, dy, ddy, kappa, dkappa, angle = self.get_point(x_scat[i])
                dx = x - points[-1].x
                dy = y - points[-1].y
                s = points[-1].s + np.sqrt(dx**2 + dy**2)
                points.append(Point(x, y, dy, ddy, kappa, dkappa, s, angle))
        return points

    def get_nearest_point(self, x: float, y: float):
        """
        calculate the nearest point
            Args:
                x (float): x
                y (float): y
            Returns:
                nearest_point (Point)
        """
        min_dist = np.inf
        nearest_point = Point(0, 0, 0, 0, 0, 0, 0, 0)
        for point in self.points:
            dist = np.sqrt((point.x - x) ** 2 + (point.y - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_point = point
        return nearest_point

    def get_point_from_S(self, nearest_point: Point, ds: float):
        """
        根据s值，返回参考线点。

            Args:
                nearest_point (Point): 最近点
                ds (float): s值

            Returns:
                x, y, dy, ddy, kappa, dkappa, angle
        """
        s = nearest_point.s + ds
        for i in range(len(self.points)):
            if self.points[i].s > s:
                rho_s = self.points[i].s - s
                x = self.points[i].x + rho_s * (self.points[i].x - self.points[i - 1].x)
                y = self.points[i].y + rho_s * (self.points[i].y - self.points[i - 1].y)
                dy = self.points[i].dy + rho_s * (
                    self.points[i].dy - self.points[i - 1].dy
                )
                ddy = self.points[i].ddy + rho_s * (
                    self.points[i].ddy - self.points[i - 1].ddy
                )
                kappa = self.points[i].kappa + rho_s * (
                    self.points[i].kappa - self.points[i - 1].kappa
                )
                dkappa = self.points[i].dkappa + rho_s * (
                    self.points[i].dkappa - self.points[i - 1].dkappa
                )
                angle = self.points[i].angle + rho_s * (
                    self.points[i].angle - self.points[i - 1].angle
                )
                return Point(x, y, dy, ddy, kappa, dkappa, s, angle)

    def get_inline_pointY_frm_x(self, x: float):
        return self.a0 + self.a1 * x + self.a2 * x**2

    def get_vehicle_status_from_settings(
        self, pos_left: bool, yaw_left: bool, kappa_left: bool
    ) -> vehicle_status:
        x = 0.0  # vehicle start at 0 as default
        x, y, dy, ddy, kappa, dkappa, angle = self.get_point(x)
        pos_coeff = 1.0 if pos_left else -1.0
        yaw_coeff = 1.0 if yaw_left else -1.0
        kappa_coeff = 1.0 if kappa_left else -1.0
        return vehicle_status(
            x - MAX_Dist_Diff * pos_coeff * cos(angle),
            y - MAX_Dist_Diff * pos_coeff * sin(angle),
            angle + MAX_Heading_Diff * yaw_coeff,
            kappa + MAX_Heading_Diff * kappa_coeff,
            max_velocity_vs_kappa(kappa),
            0.0
        )
mid =  (field_size["y_max"] + field_size["y_min"]) / 2

straight_road = reference_line(mid, 0.0, 0.0)

left_curve_road = reference_line(mid, 0.05, 0.01)

right_curve_road = reference_line(mid, -0.05, -0.01)

# do some test
if __name__ == "__main__":

    preview_dt = 20.0  # s
    velocity = 10.0

    preview_dist = velocity * preview_dt
    ref_line = reference_line(10.0, 0.005, 0.0005)
    plan_traj = ref_line.get_ref_points(preview_dist)
    X = [p.x for p in plan_traj]
    Y = [p.y for p in plan_traj]
    plt.xlim(-5, 200)
    plt.ylim(-5, 100)
    plt.scatter(X, Y, s=1, c="r")
    plt.show()
