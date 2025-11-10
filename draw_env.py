from vehicle_model import vehicle_model
import matplotlib.pyplot as plt
from referenceline import reference_line
from controller import LongPid_Controller, LatKmMpc_Controller, ts, horizon
from object import object, target_sensor
from utilities import *

############################################################################
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |# '.
#                 / \\|||  :  |||# \
#                / _||||| -:- |||||- \
#               |   | \\\  -  #/ |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
###########################################################################


if __name__ == "__main__":

    #########objects creation#####
    ref_lin = reference_line(10.0, 0.005, 0.002)  # create a reference line
    #########initialize##########

    ego = vehicle_model("ego", 0.01, 0.002, 15.0, -4, 0, 20)  # create a vehicle model
    sensor = target_sensor(ego)
    sensor.register(object("car", 1.9, 5.0, 20.0, 15.0, ref_lin, 2.0))
    trajectory = ref_lin.get_ref_points(200)  # get reference line
    lat_controller = LatKmMpc_Controller(ts, horizon)  # create a lateral controller
    lon_controller = LongPid_Controller(1.5, 30.0)
    #########figure setup#########
    fig, ax = plt.subplots()
    #fig.canvas.manager.
    fig.tight_layout()
    ax.set_facecolor("lightgreen")
    ax.set_xlim(-10, 200)
    ax.set_ylim(-10, 100)
    plt.ion()  # 开启 交互模式
    show_start_message(ax=ax)
    plt.pause(2.0)
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

        plt.scatter(traj_x, traj_y, s=1, c="r")  # draw reference line in global frame
        show_time(ax= ax, loc_x= 0.05, loc_y= 0.95, time= i * ts) # show time
        show_info(ax= ax, loc_x= 0.05, loc_y= 80, info= "velocity : " + str(ego.velocity) + " m/s \n" +
                  "acceleration : " + str(ego.acceleration) + " m/s^2 \n" + 
                  "kappa : " + str(ego.kappa) + " 1/m \n")
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
        #### update controller and get control command #####
        kappa_rate = lat_controller.Update(ego.get_vehicle_status(), control_ref)
        acceleration = lon_controller.Update(ego, sensor.get_object_by_name("car"))
        print(
            "kappa_rate : ", kappa_rate, " acceleration : ", acceleration
        )  # print control command

        #### plot control reference point #####
        control_ref_x = [pt.x for pt in control_ref]
        control_ref_y = [pt.y for pt in control_ref]
        plt.scatter(control_ref_x, control_ref_y, s=10, c="b")

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
    plt.show()
