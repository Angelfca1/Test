from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import math
import numpy as np


def forward_kinematics(l1, l2, theta1, theta2):
    c1 = math.cos(theta1)
    c2 = math.cos(theta2)
    s1 = math.sin(theta1)
    s2 = math.sin(theta2)
    O01 = [0, 0]
    O12 = [l1, 0]

    H01 = np.array([[c1, -s1, O01[0]], [s1, c1, O01[1]], [0, 0, 1]])
    H12 = np.array([[c2, -s2, O12[0]], [s2, c2, O12[1]], [0, 0, 1]])
    H02 = np.matmul(H01, H12)
    o = [0, 0]

    P1 = np.array([l1, 0, 1])
    P1 = np.transpose(P1)
    P0 = np.matmul(H01, P1)
    p = [P0[0], P0[1]]

    Q2 = np.array([l2, 0, 1])
    Q2 = np.transpose(Q2)
    Q0 = np.matmul(H02, Q2)
    q = [Q0[0], Q0[1]]

    return o, p, q


def inverse_kinematics(theta, parms):
    l1 = parms[0]
    l2 = parms[1]
    x_ref = parms[2]
    y_ref = parms[3]

    theta1 = theta[0]
    theta2 = theta[1]
    [o, p, q] = forward_kinematics(l1, l2, theta1, theta2)
    x = q[0]
    y = q[1]
    return x - x_ref, y - y_ref


l1 = 1
l2 = 1
# phi = np.arange(0, 2 * np.pi, 0.1)
phi = []
x_ref = []
y_ref = []
for i in range(0, 1000):
    phic = 2 * np.pi * i
    xcalc = 0.8 + 0.5 * np.cos((phic + (2 * np.pi)) / 5)
    ycalc = 0.8 + 0.5 * np.sin((phic + (2 * np.pi)) / 5)
    x_ref.append(xcalc)
    y_ref.append(ycalc)
    phi.append(phic)

# x_ref = 1 + 0.5 * np.cos(phi+(2*np.pi))
# y_ref = 0.5 + 0.5 * np.sin(phi+(2*np.pi))

theta1_all = []
theta2_all = []

for i in range(0, len(phi)):
    parms = [l1, l2, x_ref[i], y_ref[i]]
    theta = fsolve(inverse_kinematics, [0.01, 0.5], parms)
    theta1_all.append(theta[0])
    theta2_all.append(theta[1])

for i in range(len(phi)):
    [o, p, q] = forward_kinematics(l1, l2, theta1_all[i], theta2_all[i])
    plt.plot(q[0], q[1], color='black', marker='o', markersize=5)
    plt.plot(x_ref, y_ref, color='blue', marker='o', markersize=5)

    tmp1, = plt.plot([o[0], p[0]], [o[1], p[1]], linewidth=5, color='red')
    tmp2, = plt.plot([p[0], q[0]], [p[1], q[1]], linewidth=5, color='blue')
    plt.xlabel("x")
    plt.ylabel("y")

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal')

    plt.pause(0.2)
    tmp1.remove()
    tmp2.remove()

plt.close()
