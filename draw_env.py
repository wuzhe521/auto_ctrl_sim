from vehicle_model import vehicle_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from referenceline import reference_line


def Coordinate_transform(x, y, x0, y0, angle):
    rotate_x = (x) * np.cos(angle) + (y) * np.sin(angle)
    rotate_y = -(x) * np.sin(angle) + (y) * np.cos(angle)
    tran_x = rotate_x + x0
    tran_y = rotate_y + y0
    return tran_x, tran_y


if __name__ == "__main__":

    preview_t = 10.0  # Planner time horizon
    control_ts = 0.2  # s
    horizon = 10      # horizon length
    E0Y = vehicle_model("E0Y", 0.0, 0.005, 10.0, 0.5, -5, -5)
    ref_lin = reference_line(-5.0, 0.005, 0.0005)
    trajectory = ref_lin.get_ref_points(preview_t * E0Y.velocity)
    print(ref_lin.points[0])
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 150)
    ax.set_ylim(-10, 10)
    traj_x = [p.x for p in trajectory]
    traj_y = [p.y for p in trajectory]

    plt.ion()  # 开启 交互模式

    for i in range(20):
        # set x-axis from -10 to 10
        ax.set_xlim(-10, 100)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        preview_s = E0Y.velocity * preview_t

        rect = patches.Polygon(
            E0Y.position(),
            linewidth=2,
            edgecolor="blue",
            facecolor="lightblue",
            alpha=0.7,
        )
        ax.add_patch(rect)
        nearest_point = ref_lin.get_nearest_point(E0Y.X, E0Y.Y)  # get nearest point
        ds = [E0Y.velocity * control_ts * i for i in range(horizon)]
        # print(ds)
        control_ref = []
        for i in range(horizon):
            control_ref.append(
                ref_lin.get_point_from_S(nearest_point, ds[i])
            )
        control_ref_x = [pt.x for pt in control_ref]
        control_ref_y = [pt.y for pt in control_ref]
        
        plt.scatter(control_ref_x, control_ref_y, s=5, c="b")
        plt.scatter(traj_x, traj_y, s=1, c="r")  # draw reference line
        plt.scatter(nearest_point.x, nearest_point.y, s=5, c="g")  # draw nearest point
        E0Y.kinematic_Update(0, 0.2)  # update kinematic model
        plt.pause(0.5)
        ax.cla()
    # 关闭交互模式
    plt.ioff()
    ax.set_xlim(-10, 100)
    ax.set_ylim(-10, 10)
    plt.show()
