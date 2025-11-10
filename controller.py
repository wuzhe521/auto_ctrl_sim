import numpy as np
import matplotlib.pyplot as plt
from vehicle_model import vehicle_status, vehicle_model
from referenceline import reference_line
from math import *
import osqp
import scipy.sparse as sp
from object import object

ts = 0.2  # sample time
horizon = 10  # horizon length

max_jerk = 0.5
max_kappa_rate = 0.05

max_acc = 2.0

state_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
state_max = np.array([np.inf, np.inf, np.inf, np.inf])


class LatKmMpc_Controller:
    def __init__(self, ts: float, horizon: int):
        self.ts = ts
        self.horizon = horizon
        # weight matrix
        self.Q = sp.csc_array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 100.0, 0.0],
                [0.0, 0.0, 0.0, 2000.0],
            ]
        )
        self.R = 10.0
        self.QN = self.Q
        # reference
        self.ref = []
        # initial status
        self.init_status = vehicle_status(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def Update(self, init_status: vehicle_status, trajectory) -> float:
        """
        Calculate the control command using Kinematic model and OSQP.

        more details:

            https://github.com/osqp/osqp

        input:
            init_status: initial status
            trajectory: reference trajectory

        output:
            control command[ kappa rate]
        """
        self.init_status = init_status
        self.ref = trajectory  # update reference
        x0 = init_status.x
        y0 = init_status.y
        theta0 = init_status.theta
        d_theta0 = init_status.theta - self.ref[0].angle
        kappa0 = init_status.kappa
        v0 = init_status.velocity

        state0 = np.array([x0, y0, d_theta0, kappa0])  # initial state
        W_0 = np.array([v0 * np.cos(theta0), v0 * np.sin(theta0), 0.0, 0.0])
        Ad = sp.csc_matrix(
            [
                [1.0, 0.0, -v0 * sin(theta0) * ts, 0.0],
                [0.0, 1.0, v0 * cos(theta0) * ts, 0.0],
                [0.0, 0.0, 1.0, v0 * ts],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        Bd = sp.csc_matrix([[0.0], [0.0], [0.0], [1.0]])
        Wd = sp.csc_matrix(
            [v0 * np.cos(theta0) * ts, v0 * np.sin(theta0) * ts, 0.0, 0.0]
        )
        [nx, nu] = Bd.shape
        u_lower = -max_kappa_rate
        u_upper = max_kappa_rate
        # construct the cost
        P = sp.block_diag(
            [
                sp.kron(sp.eye(horizon), self.Q),
                self.QN,
                sp.kron(sp.eye(horizon), self.R),
            ],
            format="csc",
        )
        q = []
        for i in range(horizon):  # 0 -> N-1
            xr = trajectory[i].x
            yr = trajectory[i].y
            dl_theta = trajectory[i].angle - trajectory[0].angle
            kappa_r = trajectory[i].kappa
            state_r = np.array([xr, yr, dl_theta, kappa_r])
            q = np.hstack([q, -self.Q @ state_r])
        # print("part 1 q size:", q.shape)
        state_rn = np.array(
            [
                self.ref[-1].x,
                self.ref[-1].y,
                self.ref[-1].angle - trajectory[0].angle,
                self.ref[-1].kappa,
            ]
        )

        q = np.hstack([q, -self.QN @ state_rn, np.zeros(horizon * nu)])
        # print("part 2 q size:", q.shape)
        # - linear equations
        Ax = sp.kron(sp.eye(horizon + 1), -sp.eye(nx)) + sp.kron(
            sp.eye(horizon + 1, k=-1), Ad
        )
        Bu = sp.kron(sp.vstack([sp.csc_matrix((1, horizon)), sp.eye(horizon)]), Bd)
        Aeq = sp.hstack([Ax, Bu])
        beq = np.zeros(horizon * nx + horizon * nu)
        leq = np.hstack([-state0, np.zeros(horizon * nx)]) - np.kron(
            np.ones(horizon + 1), W_0
        )
        ueq = leq
        # - inequality constraints
        Aineq = sp.eye((horizon + 1) * nx + horizon)
        lineq = np.hstack(
            [
                np.kron(np.ones(horizon + 1), state_min),
                np.kron(np.ones(horizon), u_lower),
            ]
        )
        uineq = np.hstack(
            [
                np.kron(np.ones(horizon + 1), state_max),
                np.kron(np.ones(horizon), u_upper),
            ]
        )
        # print("Aeq size:", Aeq.shape, "Aineq size:", Aineq.shape, "lineq size:", lineq.shape, "uineq size:", uineq.shape)
        # OSQP constraints
        A = sp.vstack([Aeq, Aineq])
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])
        # Create an OSQP object
        prob = osqp.OSQP()

        prob.setup(P, q, A, l, u, warm_start=True, verbose=False)
        res = prob.solve()
        # get control
        ctrl = res.x[-horizon * nu : -(horizon - 1) * nu]

        return ctrl[0]

    def get_ref_points(self, ref_points: list):
        """
        get reference points
            input
                ref_points: reference points
        """
        self.ref = ref_points
        return True


class LongPid_Controller:
    def __init__(self, headway: float, set_speed: int):

        ### follow PID ###
        self.P_pos = 0.5
        self.P_vel = 1.0
        ### speed PID ###
        self.P_spd = 1.0
        ### headway ###
        self.headway = headway  # second
        ### set speed ###
        self.set_speed = set_speed
        ### limit ###
        self.acc_max = 2.0
        self.acc_min = -2.0

    def Update(self, ego_vehicle: vehicle_model, target_vehicle: object):
        target_s_gap = target_vehicle.velocity * self.headway
        current_s_gap = target_vehicle.s - ego_vehicle.s
        s_gap = current_s_gap - target_s_gap
        v_gap = target_vehicle.velocity - ego_vehicle.velocity
        ### set speed control ###
        set_speed_a = self.P_spd * (self.set_speed - ego_vehicle.velocity)
        set_speed_a = max(self.acc_min, min(self.acc_max, set_speed_a))
        ### follow target vehicle ###
        outer_pid = self.P_pos * s_gap
        inner_pid = self.P_vel * (v_gap + outer_pid)
        lead_target_a = max(min(inner_pid, self.acc_max), self.acc_min)
        return min(set_speed_a, lead_target_a)


if __name__ == "__main__":
    planner_t = 10.0
    ego = vehicle_model("ego", 0.0, 0.005, 10.0, 0.5, -5.0, -5.0)
    ref_lin = reference_line(-5.0, 0.005, 0.0005)
    trajectory = ref_lin.get_ref_points(planner_t * ego.velocity)

    controller = LatKmMpc_Controller(ts, horizon)
    nearest_point = ref_lin.get_nearest_point(ego.X, ego.Y)

    ts = controller.ts
    horizon = controller.horizon
    ds = [ego.velocity * ts * i for i in range(horizon)]
    control_ref = []
    for i in range(horizon):
        control_ref.append(ref_lin.get_point_from_S(nearest_point, ds[i]))
    print(controller.Update(ego.get_vehicle_status(), control_ref))

    print("pause")
