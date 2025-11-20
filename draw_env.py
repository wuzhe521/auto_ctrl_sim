from vehicle_model import vehicle_model, vehicle_status
import matplotlib.pyplot as plt
from referenceline import reference_line, straight_road, left_curve_road, right_curve_road
from controller import LongPid_Controller, LatKmMpc_Controller, ts, horizon
from object import object, target_sensor
from utilities import *
from typing import List, Dict


if __name__ == "__main__":

    #########objects creation#####
    ref_lin = straight_road  # create a reference line
    #########initialize##########

    ego = vehicle_model("ego", 0.01, 0.002, 15.0, -4, 0, 40)  # create a vehicle model
    sensor = target_sensor(ego)
    sensor.register(object("car", 1.9, 5.0, 20.0,100.0/3.6, ref_lin, 2.0))
    trajectory = ref_lin.get_ref_points(field_size["x_max"])  # get reference line
    lat_controller = LatKmMpc_Controller(ts, horizon)  # create a lateral controller
    lon_controller = LongPid_Controller(1.5, 30.0)
    #########figure setup#########
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Spark Group Simulation Platform")
    #fig.canvas.manager.
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
    Hist_Sts : List[vehicle_status] = []
    Hist_Cmd : Dict[str, list] = {"kappa_cmd":[], "accel_cmd":[]}
    time : List[float] = []
    ####### simulation loop ######
    for i in range(50):
        # set x-axis from -10 to 10
        ax.set_xlim(field_size["x_min"], field_size["x_max"])
        ax.set_ylim(field_size["y_min"], field_size["y_max"])

        plt.scatter(traj_x, traj_y, s=1, c="r")  # draw reference line in global frame
        show_time(ax= ax, loc_x= 0.05, loc_y= 0.95, time= i * ts) # show time
        show_info(ax= ax, loc_x= 0.05, loc_y= 80, info= f"velocity : {ego.velocity: 0.2e}, m/s \n" +
                  f"acceleration :  {ego.acceleration: 0.2e} m/s^2 \n" +  
                  f"kappa :   {ego.kappa:0.2e} 1/m \n")
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
            ref_lin.get_point_from_S(nearest_point, ds[i])
            for i in range(horizon)
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
        
        Hist_Sts.append(ego_status)
        Hist_Cmd["kappa_cmd"].append(kappa_rate)
        Hist_Cmd["accel_cmd"].append(acceleration)
        time.append(i * ts) 
        ##### kinematic model update #####
        ego.kinematic_Update(
            kappa_rate=kappa_rate, acceleration=acceleration, dt=ts
        )  # update kinematic model
        #### sensor update #####
        sensor.Update(ts)

        #### pause for visualization #####
        plt.pause(0.1)
        ax.cla()  # 清空画布
    # 关闭交互模式
    plt.ioff()
    ax.set_xlim(-10, 200)
    ax.set_ylim(-10, 100)
    show_end_message(ax)
    
    #####################################
    #########  Show Info History ########
    #####################################
    fig2, axes = plt.subplots(5, 1)
    fig2.canvas.manager.set_window_title( "Ego Vehicle Motion Info")
    kappa = [item.kappa for item in Hist_Sts]
    axes[0].plot(time, kappa, c = 'r', label = 'kappa')
    axes[0].set_ylabel('$\kappa$')
    axes[0].grid(True)
    velocity = [item.velocity for item in Hist_Sts]
    axes[1].plot(time, velocity, c = 'b', label = 'velocity')
    axes[1].set_ylabel('$v$')
    axes[1].grid(True)
    acceleration = [item.acceleration for item in Hist_Sts]
    axes[2].plot(time, acceleration, c = 'g', label = 'acceleration')
    axes[2].set_ylabel('ax')
    axes[2].grid(True)
    
    axes[3].plot(time, Hist_Cmd["kappa_cmd"], c = 'y', label = 'kr_cmd')
    axes[3].set_ylabel('kappa_cmd')
    axes[3].grid(True)
    
    axes[4].plot(time, Hist_Cmd["accel_cmd"], c = 'purple', label = 'acc_cmd')
    axes[4].set_ylabel('acc_cmd')
    axes[4].grid(True)

    plt.show()
    
