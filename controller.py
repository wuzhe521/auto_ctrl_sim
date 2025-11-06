import numpy as np
import matplotlib.pyplot as plt
from vehicle_model import vehicle_status, vehicle_model
from referenceline import reference_line
from math import *
import osqp
import scipy.sparse as sp


ts = 0.2  # sample time
horizon = 10  # horizon length

max_jerk = 0.5
max_kappa_rate = 0.05
state_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
state_max = np.array([np.inf, np.inf, np.inf, np.inf])

class MPC_Controller:
    def __init__(self, ts: float, horizon: int):
        self.ts = ts
        self.horizon = horizon
        # weight matrix
        self.Q = sp.eye(4)
        self.R = 1
        self.QN = self.Q
        # reference
        self.ref = []
        # initial status
        self.init_status = vehicle_status(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def Update(self, init_status: vehicle_status, trajectory):
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
        for i in range(horizon): # 0 -> N-1
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
        leq = np.hstack([-state0, np.zeros(horizon * nx)]) - np.kron(np.ones(horizon + 1), W_0)
        ueq = leq
        # - inequality constraints
        Aineq = sp.eye((horizon + 1) * nx + horizon)
        lineq = np.hstack(
            [np.kron(np.ones(horizon + 1), state_min), np.kron(np.ones(horizon), u_lower)]
        )
        uineq = np.hstack(
            [np.kron(np.ones(horizon + 1), state_max), np.kron(np.ones(horizon), u_upper)]
        )
        # print("Aeq size:", Aeq.shape, "Aineq size:", Aineq.shape, "lineq size:", lineq.shape, "uineq size:", uineq.shape)
        # OSQP constraints
        A = sp.vstack([Aeq, Aineq])
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])
        # Create an OSQP object
        prob = osqp.OSQP()

        prob.setup(P, q, A, l, u, warm_starting=True, verbose=False)
        res = prob.solve()
        # get control
        ctrl = res.x[-horizon * nu : -(horizon - 1) * nu]

        return ctrl[0]

    def get_ref_points(self, ref_points: list):
        self.ref = ref_points
        return True


if __name__ == "__main__":
    planner_t = 10.0
    E0Y = vehicle_model("E0Y", 0.0, 0.005, 10.0, 0.5, -5.0, -5.0)
    ref_lin = reference_line(-5.0, 0.005, 0.0005)
    trajectory = ref_lin.get_ref_points(planner_t * E0Y.velocity)

    controller = MPC_Controller(ts, horizon)
    nearest_point = ref_lin.get_nearest_point(E0Y.X, E0Y.Y)

    ts = controller.ts
    horizon = controller.horizon
    ds = [E0Y.velocity * ts * i for i in range(horizon)]
    control_ref = []
    for i in range(horizon):
        control_ref.append(ref_lin.get_point_from_S(nearest_point, ds[i]))
    print(controller.Update(E0Y.get_vehicle_status(), control_ref))
    
    print("pause")
    
