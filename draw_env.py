from vehicle_model import vehicle_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from referenceline import reference_line
from controller import MPC_Controller, ts, horizon


def Coordinate_transform(x, y, x0, y0, angle):
    rotate_x = (x) * np.cos(angle) + (y) * np.sin(angle)
    rotate_y = -(x) * np.sin(angle) + (y) * np.cos(angle)
    tran_x = rotate_x + x0
    tran_y = rotate_y + y0
    return tran_x, tran_y


if __name__ == "__main__":
    ########sim settings##########
    preview_t = 10.0  # Planner time horizon
    control_ts = 0.2  # s
    horizon = 10  # horizon length

    #########objects creation#####
    E0Y = vehicle_model("E0Y", 0.0, 0.005, 10.0, 0.5, 50, 20)  # create a vehicle model
    ref_lin = reference_line(-5.0, 0.005, 0.005)  # create a reference line
    trajectory = ref_lin.get_ref_points(200)  # get reference line
    controller = MPC_Controller(ts, horizon)  # create a controller

    #########figure setup#########
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.set_xlim(-10, 200)
    ax.set_ylim(-10, 100)
    plt.ion()  # 开启 交互模式
    ax.text(
        0.5,
        0.5,
        "Active Safty - LSS \n Sim Start",
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        size=18,
        fontdict=None,
    )
    plt.pause(1.0)
    ax.cla()
    #########trajectory##########
    traj_x = [p.x for p in trajectory]
    traj_y = [p.y for p in trajectory]
    #############################

    ####### simulation loop######
    for i in range(50):
        # set x-axis from -10 to 10
        ax.set_xlim(-10, 200)
        ax.set_ylim(-10, 100)
        ax.set_aspect("equal")
        plt.scatter(traj_x, traj_y, s=1, c="r")  # draw reference line in global frame

        ######## plot vehicle #####
        loc = E0Y.position()
        rect = patches.Polygon(
            loc,
            linewidth=2,
            edgecolor="blue",
            facecolor="lightblue",
            alpha=0.7,
        )

        ax.add_patch(rect)

        #### find nearest point ######
        nearest_point = ref_lin.get_nearest_point(E0Y.X, E0Y.Y)  # get nearest point
        plt.scatter(nearest_point.x, nearest_point.y, s=5, c="g")  # draw nearest point

        #### get control reference point #####
        ds = [E0Y.velocity * ts * i for i in range(horizon)]
        control_ref = []
        for i in range(horizon):
            control_ref.append(ref_lin.get_point_from_S(nearest_point, ds[i]))

        #### update controller and get control command #####
        ctrl = controller.Update(E0Y.get_vehicle_status(), control_ref)
        print("Control : ", ctrl)  # print control command

        #### plot control reference point #####
        control_ref_x = [pt.x for pt in control_ref]
        control_ref_y = [pt.y for pt in control_ref]
        plt.scatter(control_ref_x, control_ref_y, s=10, c="b")

        ##### kinematic model update #####
        E0Y.kinematic_Update(ctrl, ts)  # update kinematic model
        # E0Y.kinematic_Update(0, ts)

        #### pause for visualization #####
        plt.pause(0.1)
        ax.cla()  # 清空画布
    # 关闭交互模式
    plt.ioff()
    ax.set_xlim(-10, 200)
    ax.set_ylim(-10, 100)
    ax.text(
        0.5,
        0.5,
        "Active Safty -  LSS \n Sim End \n Thank you for using Ctrl_Sim",
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
        size=18,
        fontdict=None,
    )
    plt.show()
