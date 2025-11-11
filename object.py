import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from referenceline import reference_line, Point
from vehicle_model import vehicle_model
from math import *
from typing import List

class object:
    def __init__(
        self,
        name: str,
        Width: float,
        Length: float,
        s0: float,
        velocity: float,
        ref_line: reference_line,
        offset: float = 0.0
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
        self.loc.x  -= self.offset * cos(self.loc.angle)
        self.loc.y  += self.offset * sin(self.loc.angle)
        
    def position(self):
        """
        get vehicle 2D position in global coordinate
            output:
                points: 4 corner points of vehicle in global coordinate
        """
        angle = self.loc.angle
        X = self.loc.x
        Y = self.loc.y
        left_front_x = X + self.Length * np.cos(angle) - self.Width / 2 * np.sin(angle)
        left_front_y = Y + self.Length * np.sin(angle) + self.Width / 2 * np.cos(angle)
        left_front = (left_front_x, left_front_y)
        right_front_x = X + self.Length * np.cos(angle) + self.Width / 2 * np.sin(angle)
        right_front_y = Y + self.Length * np.sin(angle) - self.Width / 2 * np.cos(angle)
        right_front = (right_front_x, right_front_y)
        left_rear_x = X - self.Width / 2 * np.sin(angle)
        left_rear_y = Y + self.Width / 2 * np.cos(angle)
        left_rear = (left_rear_x, left_rear_y)
        right_rear_x = X + self.Width / 2 * np.sin(angle)
        right_rear_y = Y - self.Width / 2 * np.cos(angle)
        right_rear = (right_rear_x, right_rear_y)
        return [left_front, right_front, right_rear, left_rear]
        
    def show_object(self, ax):
        loc = self.position()
        rect = patches.Polygon(
            loc, linewidth=2, edgecolor="red", facecolor="red", alpha=0.7
        )
        ax.add_patch(rect)

class target_sensor:
    def __init__(self, ego_vehicle: vehicle_model):
        self.Object_list : List[object] = []
        self.ego_ = ego_vehicle
        
    def register(self, target : object):
        self.Object_list.append(target)
        return True
        
    def Update(self, ts : float):
        if len(self.Object_list) > 0:
            for i in range(len(self.Object_list)):
                self.Object_list[i].Update(ts)
                
    def get_object_by_name(self, name : str) -> object:
        return next((Object for Object in self.Object_list if Object.name == name), None)
        
    def plot_targets(self, ax):
        for i in range(len(self.Object_list)):
            loc_target = self.Object_list[i].show_object(ax)
            
if __name__ == "__main__":

    reference = reference_line(a0=10, a1=0.05, a2=0.002)
    forward_object = object(
        name="forward", Width=3, Length=5, s0=10, velocity=20, ref_line=reference
    )

    trajectory_x = [pt.x for pt in reference.points]
    trajectory_y = [pt.y for pt in reference.points]
    fig, ax = plt.subplots()
    # set x-axis from -10 to 10
    ax.set_xlim(-10, 100)
    ax.set_ylim(-10, 100)
    ax.set_aspect("equal")
    plt.ion()  # 开启 交互模式
    for _ in range(20):
        ax.set_xlim(-10, 200)
        ax.set_ylim(-10, 100)
        ax.set_aspect("equal")
        loc = forward_object.position()
        rect = patches.Polygon(
            loc, linewidth=2, edgecolor="blue", facecolor="lightblue", alpha=0.7
        )
        plt.scatter(trajectory_x, trajectory_y, s=2, color="b")
        ax.add_patch(rect)
        # Update
        forward_object.Update(0.2)
        plt.pause(0.2)
        ax.cla()
    # 关闭交互模式
    plt.ioff()
    ax.set_xlim(-10, 200)
    ax.set_ylim(-10, 100)
    ax.set_aspect("equal")
    plt.show()
