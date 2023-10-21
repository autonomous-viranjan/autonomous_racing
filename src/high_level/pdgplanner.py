#########################################
# Viranjan Bhattacharyya
# Nov 2022
#########################################

import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
import time

import sys
from ilqr import iLQR
from ilqr.cost import QRCost
from ilqr.dynamics import AutoDiffDynamics


class PDGILQR():

    def __init__(self, dt, Horizon, dsafe):        
        
        self.dt = dt
        self.N = Horizon
        self.dsafe = dsafe

    def on_iteration(self, iteration_count, xs, us, J_opt, accepted, converged):
        J_hist = []
        J_hist.append(J_opt)
        info = "converged" if converged else ("accepted" if accepted else "failed")
        print("iteration", iteration_count, info, J_opt)

    def dynamics(self):        
        """ 
        Unicycle model  for each agent i
        Xi = [xi, yi, vi, thetai]'
        Ui = [ai, omegai]'

        States and controls defined as: 
        x = [Xi Xj Goali Ggoalj dsafe Ui Uj]'
        u = [delta_ui delta_uj]'

        """
        
        x = [
            T.dscalar("x1"),
            T.dscalar("y1"),
            T.dscalar("v1"),
            T.dscalar("theta1"),    
            T.dscalar("x2"),
            T.dscalar("y2"),
            T.dscalar("v2"),
            T.dscalar("theta2"),
            T.dscalar("rx1"),
            T.dscalar("ry1"),
            T.dscalar("rx2"),
            T.dscalar("ry2"),
            T.dscalar("dsafe"),
            T.dscalar("a1"),
            T.dscalar("omega1"),
            T.dscalar("a2"),
            T.dscalar("omega2")
        ]

        u = [
            T.dscalar("d_a1"),
            T.dscalar("d_omega1"),
            T.dscalar("d_a2"),
            T.dscalar("d_omega2")
        ]

        # Define discrete time dynamics
        f = T.stack([
            x[0] + x[2] * np.cos(x[3]) * self.dt,
            x[1] + x[2] * np.sin(x[3]) * self.dt,
            x[2] + x[13] * self.dt,
            x[3] + x[14] * self.dt,    
            x[4] + x[6] * np.cos(x[7]) * self.dt,
            x[5] + x[6] * np.sin(x[7]) * self.dt,
            x[6] + x[15] * self.dt,
            x[7] + x[16] * self.dt,
            x[8],
            x[9],
            x[10],
            x[11],
            x[12],
            x[13] + u[0],
            x[14] + u[1],
            x[15] + u[2],
            x[16] + u[3]
        ])

        dynamics = AutoDiffDynamics(f, x, u)

        return dynamics

    def cost(self):
        """ 
        min J = J_running + J_terminal

        J_running = {(x1 - x2 - dsafe)^2 + (y1 - y2 - dsafe)^2}
        J_terminal = {qx1 * [x1 - xgoal1]^2 + qy1 * [y1 - ygoal1]^2}
                    + {qx2 * [x2 - xgoal2]^2 + qy2 * [y2 - ygoal2]^2}
                    + {(x1 - x2 - dsafe)^2 + (y1 - y2 - dsafe)^2}

        X = [X1 X2]
        X_augmented = [X xgoal1 ygoal1 xgoal2 ygoal2 dsafe a1 omega1 a2 ommega2]'
        U = [delta_a1 delta_omega1 delta_a2 delta_omega2]'

        => J = X_augmented' Q X_augmented + U' R U
        """
        nx = self.dynamics().state_size
        nu = self.dynamics().action_size

        Q = np.zeros((nx, nx))        
        # Only collision avoidance in running cost
        Q[0, 0] = 1
        Q[0, 4] = -1
        Q[1, 1] =  1
        Q[1, 5] = -1
        Q[4, 0] = -1
        Q[4, 4] = 1
        Q[5, 1] = -1
        Q[5, 5] = 1
        Q[12, 0] = -2
        Q[12, 1] = -2
        Q[12, 4] = 2
        Q[12, 5] = 2
        Q[12, 12] = 2

        Q_T = np.zeros((nx, nx))
        # Goal tracking and collision avoidance in terminal cost
        Q_T[0, 0] = 2
        Q_T[0, 4] = -1
        Q_T[0, 8] = -1
        Q_T[1, 1] = 2
        Q_T[1, 5] = -1
        Q_T[1, 9] = -1
        Q_T[4, 0] = -1
        Q_T[4, 4] = 2
        Q_T[4, 10] = -1
        Q_T[5, 1] = -1
        Q_T[5, 5] = 2
        Q_T[5, 11] = -1
        Q_T[8, 0] = -1
        Q_T[8, 8] = 1
        Q_T[9, 1] = -1
        Q_T[9, 9] = 1
        Q_T[10, 4] = -1
        Q_T[10, 10] = 1
        Q_T[11, 5] = -1
        Q_T[11, 11] = 1
        Q_T[12, 0] = -2
        Q_T[12, 1] = -2
        Q_T[12, 4] = 2
        Q_T[12, 5] = 2
        Q_T[12, 12] = 2

        Q_running = Q
        R = np.eye(nu)
        cost = QRCost(Q_running, R, Q_terminal=Q_T)

        return cost

    def solve_ilqr(self, x1_0, y1_0, v1_0, theta1_0, x2_0, y2_0, v2_0, theta2_0, xgoal1, ygoal1, xgoal2, ygoal2, a1_0, omega1_0, a2_0, omega2_0):

        nx = self.dynamics().state_size
        nu = self.dynamics().action_size

        x0 = np.array([x1_0, y1_0, v1_0, theta1_0, \
                x2_0, y2_0, v2_0, theta2_0, \
                xgoal1, ygoal1, xgoal2, ygoal2, self.dsafe, a1_0, omega1_0, a2_0, omega2_0])  # Initial joint state.

        # initialization        
        u_init = np.zeros((self.N, nu))
        
        # J_hist = []
        ilqr = iLQR(self.dynamics(), self.cost(), self.N)
        Xsol, Usol = ilqr.fit(x0, u_init, on_iteration=self.on_iteration)

        return Xsol, Usol

# Unit Test
if __name__ == "__main__":
    dt = 0.1
    Horizon = 20
    # Initial states
    x1_0 = 0
    y1_0 = 0
    v1_0 = 0
    theta1_0 = 0
    x2_0 = 0
    y2_0 = 3
    v2_0 = 0
    theta2_0 = 0
    # Initial augemnted control-states
    a1_0 = 0
    omega1_0 = 0
    a2_0 = 0
    omega2_0 = 0
    
    # Goal
    xgoal1 = 10
    ygoal1 = 5
    xgoal2 = 10
    ygoal2 = 5
    dsafe = 0.5

    t0 = time.time()

    Xsol, Usol = PDGILQR(dt, Horizon, dsafe).solve_ilqr(x1_0, y1_0, v1_0, theta1_0, x2_0, y2_0, v2_0, theta2_0, \
                        xgoal1, ygoal1, xgoal2, ygoal2, a1_0, omega1_0, a2_0, omega2_0)

    t1 = time.time()
    total = t1-t0
    print(total)

    # Extract trajectory
    x1 = Xsol[:, 0]
    y1 = Xsol[:, 1]
    v1 = Xsol[:, 2]
    theta1 = Xsol[:, 3]
    x2 = Xsol[:, 4]
    y2 = Xsol[:, 5]
    v2 = Xsol[:, 6]
    theta2 = Xsol[:, 7]

    a1 = Xsol[:, 13]
    omega1 = Xsol[:, 14]
    a2 = Xsol[:, 15]
    omega2 = Xsol[:, 16]

    plt.figure(1)
    plt.plot(x1, y1, "r")
    plt.plot(x2, y2, "b")
    plt.plot(xgoal1, ygoal1, "rx")
    plt.plot(xgoal2, ygoal2, "bx")
    # plt.show()

    plt.figure(2)
    plt.subplot(211)
    plt.plot(v1, "r")
    plt.plot(v2, "b")
    plt.ylabel("Velocities")
    plt.subplot(212)
    plt.plot(theta1, "r")
    plt.plot(theta2, "b")
    plt.ylabel("Thetas")

    plt.figure(3)
    plt.subplot(211)
    plt.plot(a1, "r")
    plt.plot(a2, "b")
    plt.ylabel("Accelerations")
    plt.subplot(212)
    plt.plot(omega1, "r")
    plt.plot(omega2, "b")
    plt.ylabel("Omegas")

    plt.show()
