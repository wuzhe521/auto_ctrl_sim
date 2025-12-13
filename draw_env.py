from vehicle_model import vehicle_model, vehicle_status
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor, Slider
from referenceline import (
    reference_line,
    straight_road,
    left_curve_road,
    right_curve_road,
)
from controller import LongPid_Controller, LatKmMpc_Controller, ts, horizon
from object import object, detect_sensor
from utilities import *
from typing import List, Dict
from proto import sim_debug_pb2
from replay_data import sim_data_recorder, sim_data_player


######### recorder creation #######
debugger = sim_debug_pb2.sim_debug()
vehicle_state_debug = sim_debug_pb2.vehicle_state_debug()
controller_debug = sim_debug_pb2.controller_debug()
recorder = sim_data_recorder("test")
player = sim_data_player()
if __name__ == "__main__":
    #########objects creation##### 
    ref_lin = straight_road  # create a reference line
    ego = vehicle_model("ego", 0.01, 0.002, 15.0, -4, 0, 40)  # create a vehicle model
    sensor = detect_sensor(ego)  # equipment sensor in ego vehicle
    sensor.register_object(object("car", 1.9, 5.0, 20.0, 100.0 / 3.6, ref_lin, 2.0))
    sensor.register_ref_line(ref_lin)
    trajectory = ref_lin.get_ref_points(field_size["x_max"])  # get reference line

    lat_controller = LatKmMpc_Controller(ts, horizon)  # create a lateral controller
    lon_controller = LongPid_Controller(1.5, 30.0)
    #########figure setup#########
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Spark Group Simulation Platform")
    # fig.canvas.manager.
    fig.tight_layout()
    ax.set_facecolor("lightgreen")
    ax.set_xlim(field_size["x_min"], field_size["x_max"])
    ax.set_ylim(field_size["y_min"], field_size["y_max"])
    plt.ion()  # 开启 交互模式
    show_start_message(ax=ax)
    plt.pause(2.0)
    ax.cla()
    #########trajectory##########
    traj_x = [p.x for p in trajectory]
    traj_y = [p.y for p in trajectory]
    #############################
    ####### data container ######
    Hist_Sts: List[vehicle_status] = []
    Hist_Cmd: Dict[str, list] = {"k_r_cmd": [], "accel_cmd": []}
    time: List[float] = []
    ####### simulation loop ######
    for i in range(50):
        # set x-axis from -10 to 10
        ax.set_xlim(field_size["x_min"], field_size["x_max"])
        ax.set_ylim(field_size["y_min"], field_size["y_max"])

        plt.scatter(traj_x, traj_y, s=1, c="r")  # draw reference line in global frame
        show_time(ax=ax, loc_x=0.05, loc_y=0.95, time=i * ts)  # show time
        show_info(
            ax=ax,
            loc_x=0.05,
            loc_y=80,
            info=f"$v$ : {ego.velocity: 0.2e}, m/s \n"
            + f"$a$ :  {ego.acceleration: 0.2e} m/s^2 \n"
            + f"$\kappa$ :   {ego.kappa:0.2e} 1/m \n",
        )
        ######## plot vehicle #####
        ego.plot_vehicle(ax=ax)
        sensor.plot_targets(ax=ax)

        #### find nearest point ######
        nearest_point = ref_lin.get_nearest_point(ego.X, ego.Y)  # get nearest point
        plt.scatter(nearest_point.x, nearest_point.y, s=5, c="g")  # draw nearest point

        #### get control reference point #####
        ds = [ego.velocity * ts * i for i in range(horizon)]
        control_ref = []
        control_ref.extend(
            ref_lin.get_point_from_S(nearest_point, ds[i]) for i in range(horizon)
        )
        ego_status = ego.get_vehicle_status()

        #### update controller and get control command #####
        kappa_rate = lat_controller.Update(ego_status, control_ref)
        acceleration = lon_controller.Update(ego, sensor.get_object_by_name("car"))
        print(
            "kappa_rate : ", kappa_rate, " acceleration : ", acceleration
        )  # print control command

        #### plot control reference point #####
        control_ref_x = [pt.x for pt in control_ref]
        control_ref_y = [pt.y for pt in control_ref]
        plt.scatter(control_ref_x, control_ref_y, s=10, c="b")
        #### store ego status #####
        ego.debug_proto(debug_proto=vehicle_state_debug)
        #### store control command #####
        lat_controller.debug_proto(debug_proto=controller_debug)
        lon_controller.debug_proto(debug_proto=controller_debug)
        debugger.times.append(i * ts)
        debugger.vehicle_state_debug.append(vehicle_state_debug)    
        debugger.controller_debug.append(controller_debug)
        ##### kinematic model update #####
        ego.kinematic_Update(
            kappa_rate=kappa_rate, acceleration=acceleration, dt=ts
        )  # update kinematic model
        #### sensor update #####
        sensor.Update(ts)

        #### pause for visualization #####
        plt.pause(0.1)
        ax.cla()  # 清空画布

    plt.ioff()
    ax.set_xlim(-10, 200)
    ax.set_ylim(-10, 100)
    show_end_message(ax)
    #####################################
    ############# Save Data #############
    #####################################
    
    recorder.save_data(debugger)
    plt.close()
    player.analyze_data()



    