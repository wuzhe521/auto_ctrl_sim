import numpy as np
import matplotlib.pyplot as plt
from vehicle_model import vehicle_status
from math import *
import osqp
import scipy.sparse as sp


ts = 0.2  # sample time
horizon = 10  # horizon length

max_jerk = 0.5
max_kappa_rate = 0.05


class MPC_Controller:
    def __init__(self, ts: float, horizon: int):
        self.ts = ts
        self.horizon = horizon
        # weight matrix
        self.Q = sp.eye(4)
        self.R = sp.eye(4)
        self.QN = self.Q
        # reference
        self.ref = []
        # initial status
        self.init_status = vehicle_status(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) 
    def Update(self, init_status: vehicle_status, trajectory):
        self.init_status = init_status
        x0 = init_status.x
        y0 = init_status.y
        theta0 = init_status.theta
        kappa0 = init_status.kappa
        v0 = init_status.velocity
        self.ref = trajectory  # update reference
        state0 = np.array([x0, y0, v0, kappa0])  # initial state
        Ad = sp.csc_matrix(
            [
                [1.0, 0.0, -v0 * sin(theta0) * ts, 0.0],
                [0.0, 1.0, v0 * cos(theta0) * ts, 0.0],
                [0.0, 0.0, 1.0, v0 * ts],
            ],
            [0.0, 0.0, 0.0, 1.0],
        )
        Bd = sp.csc_matrix([0.0, 0.0, 0.0, 1.0])
        Wd = sp.csc_matrix(
            [v0 * np.cos(theta0) * ts, v0 * np.sin(theta0) * ts, 0.0, 0.0]
        )
        [nx, nu] = Bd.shape
        u_lower = -max_kappa_rate
        u_upper = max_kappa_rate
        # construct the cost
        P = sp.block_diag(
            [
                sp.kron(super.eye(horizon), self.Q),
                self.QN,
                sp.kron(sp.eye(horizon), self.R),
            ],
            format="csc",
        )
        q = np.array()
        for i in range(horizon - 1):
            xr = self.ref[i]
            q = np.hstack([q, -self.Q@xr])
        q = np.hstack([q, -self.QN@self.ref[-1], np.zeros(horizon*nu)])
        # - linear objective
        
        # - linear dynamics

        # Create an OSQP object
        prob = osqp.OSQP()

    def get_ref_points(self, ref_points : list):
        self.ref = ref_points
        return True
    
if __name__ == "__main__":
    Q = sp.eye(2)
    QN = sp.eye(2)
    R =  sp.eye(2)
    P = sp.block_diag(
        [
            sp.kron(sp.eye(horizon), Q),
            QN,
            sp.kron(sp.eye(horizon), R),
        ]
    ).toarray()
    
    print(P)
    
    xr = np.array([1.0, 0.5])
    
    q = np.hstack(
        [
            np.kron(np.ones(1), -Q @ xr),
        ]
    )
    print(q)    
    q = np.hstack([q, QN @ xr])
    print(q)