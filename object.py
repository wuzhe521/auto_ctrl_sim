import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from referenceline import reference_line, Point
from vehicle_model import vehicle_model
from math import *
from typing import List
from utilities import *

road_type = {"yellow line": 0, "white line": 1, "road edge": 2, "fence": 3}


class line_geometry:
    def __init__(self):
        self.x0 = 0.0
        self.C0 = 0.0
        self.C1 = 0.0
        self.C2 = 0.0
        self.type: int = 1


class bev_road_sensor:
    def __init__(self, name: str):
        self.name = name
        self.position = None
        self.leftleft: line_geometry = None
        self.left: line_geometry = None
        self.right: line_geometry = None
        self.rightright: line_geometry = None

    def Update(self, sensor_loc: vehicle_model, ref_line: list[reference_line]) -> None:
        global_x = sensor_loc.X
        global_y = sensor_loc.Y
        global_angle = sensor_loc.angle
        global_s = sensor_loc.s
        all_ref_left = []
        all_ref_right = []
        for item in ref_line:
            if item.detectable == False: # if not detectable skip
                continue
            nearest_point = item.get_nearest_point(global_x, global_y)
            kappa = nearest_point.kappa
            x_n, y_n, angle_n = Coordinate_transform(
                nearest_point.x,
                nearest_point.y,
                nearest_point.angle,
                global_x,
                global_y,
                global_angle,
            )
            if y_n > 0.0:
                all_ref_left.append(line_geometry(x_n, y_n, angle_n, kappa))
            else:
                all_ref_right.append(line_geometry(x_n, y_n, angle_n, kappa))
        # sort line by y
            all_ref_left = sorted(all_ref_left, key=lambda x: x.C0, reverse=True)
            all_ref_right = sorted(all_ref_right, key=lambda x: x.C0)
            self.left = all_ref_left.pop(0)
            self.leftleft = all_ref_left.pop(0)
            self.right = all_ref_right.pop(0)
            self.rightright = all_ref_right.pop(0)

class object:
    def __init__(
        self,
        name: str,
        Width: float,
        Length: float,
        s0: float,
        velocity: float,
        ref_line: reference_line,
        offset: float = 0.0,
    ):
        self.Width = Width
        self.Length = Length
        self.name = name
        self.s = s0
        self.velocity = velocity
        self.ref_l = ref_line
        self.offset = offset
        self.loc = self.ref_l.get_point_from_S(self.ref_l.points[0], self.s)

    def Update(self, ts: float):
        ds = self.velocity * ts
        self.s += ds
        self.loc = self.ref_l.get_point_from_S(self.loc, ds)
        self.loc.x -= self.offset * cos(self.loc.angle)
        self.loc.y += self.offset * sin(self.loc.angle)

    def position(self):
        """
        get vehicle 2D position in global coordinate
            output:
                points: 4 corner points of vehicle in global coordinate
        """
        # Validate required attributes
        if not hasattr(self, 'loc') or self.loc is None:
            raise AttributeError("Vehicle location (self.loc) is missing or None.")
        if not all(hasattr(self.loc, attr) for attr in ['x', 'y', 'angle']):
            raise AttributeError("Location object must have x, y, and angle attributes.")
    
        try:
            angle = self.loc.angle
            X = self.loc.x
            Y = self.loc.y
    
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
    
            half_width_sin = (self.Width / 2) * sin_angle
            half_width_cos = (self.Width / 2) * cos_angle
            length_cos = self.Length * cos_angle
            length_sin = self.Length * sin_angle
    
            left_front = (X + length_cos - half_width_sin, Y + length_sin + half_width_cos)
            right_front = (X + length_cos + half_width_sin, Y + length_sin - half_width_cos)
            left_rear = (X - half_width_sin, Y + half_width_cos)
            right_rear = (X + half_width_sin, Y - half_width_cos)
    
            return [left_front, right_front, right_rear, left_rear]
    
        except Exception as e:
            raise RuntimeError(f"Error calculating vehicle position: {str(e)}")

    def show_object(self, ax):
        loc = self.position()
        rect = patches.Polygon(
            loc, linewidth=2, edgecolor="red", facecolor="red", alpha=0.7
        )
        ax.add_patch(rect)


class detect_sensor:
    def __init__(self, ego_vehicle: vehicle_model):
        self.Object_list: List[object] = []
        self.ref_line_list: List[reference_line] = []

        self.bev_objects = []
        self.ego_ = ego_vehicle
        self.bev_road = bev_road_sensor("bev_road")
    def register_object(self, target: object):
        self.Object_list.append(target)
        return True
    def register_ref_line(self, ref_line: reference_line):
        self.ref_line_list.append(ref_line)
        return True
    def Update(self, ts: float):
        # update objects
        if len(self.Object_list) > 0:
            for i in range(len(self.Object_list)):
                self.Object_list[i].Update(ts)
        # update bev_road sensor
        if len(self.ref_line_list) > 0:
            self.bev_road.Update(self.ego_, self.ref_line_list)

    def get_object_by_name(self, name: str) -> object:
        return next(
            (Object for Object in self.Object_list if Object.name == name), None
        )

    def plot_targets(self, ax):
        for i in range(len(self.Object_list)):
            loc_target = self.Object_list[i].show_object(ax)


if __name__ == "__main__":

    reference = reference_line(a0=10, a1=0.05, a2=0.002)

    ego = vehicle_model("ego", 0.01, 0.002, 15.0, -4, 0, 40)  # create a vehicle model
    sensor = detect_sensor(ego) # equipment sensor in ego vehicle
    sensor.register_object(object("car", 1.9, 5.0, 20.0,100.0/3.6, reference, 1.7))
    sensor.register_ref_line(reference)
    trajectory = reference.get_ref_points(field_size["x_max"])  # get reference line

    trajectory_x = [pt.x for pt in reference.points]
    trajectory_y = [pt.y for pt in reference.points]
    fig, ax = plt.subplots()
    # set x-axis from -10 to 10
    ax.set_xlim(-10, 100)
    ax.set_ylim(-10, 100)
    ax.set_aspect("equal")
    plt.ion()  # 开启 交互模式
    for _ in range(30):
        ax.set_xlim(-10, 200)
        ax.set_ylim(-10, 100)
        ax.set_aspect("equal")
        sensor.plot_targets(ax=ax)
        plt.scatter(trajectory_x, trajectory_y, s=2, color="b")
        # Update
        sensor.Update(0.2)
        plt.pause(0.2)
        ax.cla()
    
    # 关闭交互模式
    plt.ioff()
    ax.set_xlim(-10, 200)
    ax.set_ylim(-10, 100)
    ax.set_aspect("equal")
    plt.show()
