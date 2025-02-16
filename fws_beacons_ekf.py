"""
Example fws_beacons_ekf.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import chi2
from mobotpy.integration import rk_four
from mobotpy.models import FourWheelSteered

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 10.0
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# MOBILE ROBOT SETUP

# Set the wheelbase and track of the vehicle [m]
ELL_W = 2.50
ELL_T = 1.75

# Let's now use the class Ackermann for plotting
vehicle = FourWheelSteered(ELL_W, ELL_T)

# %%
# CREATE A MAP OF FEATURES

# Set a minimum number of features in the map that achieves observability
N_FEATURES = 5

# Set the size [m] of a square map
D_MAP = 50.0

# Create a map of randomly placed feature locations
f_map = np.zeros((2, N_FEATURES))
for i in range(0, N_FEATURES):
    f_map[:, i] = D_MAP * (np.random.rand(2) - 0.5)

# %%
# SET UP THE NOISE COVARIANCE MATRICES

# Range sensor noise standard deviation (each feature has the same noise) [m]
SIGMA_W1 = 1.0

# Steering angle noise standard deviation [rad]
SIGMA_W2 = 0.05

# Sensor noise covariance matrix
R = np.zeros((N_FEATURES + 1, N_FEATURES + 1))
R[0:N_FEATURES, 0:N_FEATURES] = np.diag([SIGMA_W1**2] * N_FEATURES)
R[N_FEATURES, N_FEATURES] = SIGMA_W2**2

# Speed noise standard deviation [m/s]
SIGMA_U1 = 0.1

# Steering rate noise standard deviation [rad/s]
SIGMA_U2 = 0.2

# Process noise covariance matrix
Q = np.diag([SIGMA_U1**2, SIGMA_U2**2])

# %%
# FUNCTION TO MODEL RANGE TO FEATURES


def range_sensor(x, sigma_w, f_map):
    """
    Function to model the range sensor.

    Parameters
    ----------
    x : ndarray
        An array of length 2 representing the robot's position.
    sigma_w : float
        The range sensor noise standard deviation.
    f_map : ndarray
        An array of size (2, N_FEATURES) containing map feature locations.

    Returns
    -------
    ndarray
        The range (including noise) to each feature in the map.
    """

    # Compute the measured range to each feature from the current robot position
    z = np.zeros(N_FEATURES)
    for j in range(0, N_FEATURES):
        z[j] = (
            np.sqrt((f_map[0, j] - x[0]) ** 2 + (f_map[1, j] - x[1]) ** 2)
            + sigma_w * np.random.randn()
        )

    # Return the array of noisy measurements
    return z


# %%
# FUNCTION TO IMPLEMENT THE EKF FOR THE FWS MOBILE ROBOT


def fws_ekf(q, P, u, z, Q, R, f_map):
    """
    Function to implement an observer for the robot's pose.

    Parameters
    ----------
    q : ndarray
        An array of length 4 representing the robot's pose.
    P : ndarray
        The state covariance matrix.
    u : ndarray
        An array of length 2 representing the robot's inputs.
    z : ndarray
        The range measurements to the features.
    Q : ndarray
        The process noise covariance matrix.
    R : ndarray
        The sensor noise covariance matrix.
    f_map : ndarray
        An array of size (2, N_FEATURES) containing map feature locations.

    Returns
    -------
    ndarray
        The estimated state of the robot.
    ndarray
        The state covariance matrix.
    """

    # Compute the a priori estimate
    # (a) Compute the Jacobian matrices (i.e., linearize about current estimate)
    F = np.zeros((4, 4))
    F = np.eye(4) + T * np.array(
        [
            [
                0,
                0,
                -u[0] * np.sin(q[2]) * np.cos(q[3]),
                -u[0] * np.cos(q[2]) * np.sin(q[3]),
            ],
            [
                0,
                0,
                u[0] * np.cos(q[2]) * np.cos(q[3]),
                -u[0] * np.sin(q[2]) * np.sin(q[3]),
            ],
            [0, 0, 0, u[0] * 1.0 / (0.5 * ELL_W) * np.cos(q[3])],
            [0, 0, 0, 0],
        ]
    )
    G = np.zeros((4, 2))
    G = T * np.array(
        [
            [np.cos(q[2]) * np.cos(q[3]), 0],
            [np.sin(q[2]) * np.cos(q[3]), 0],
            [1.0 / (0.5 * ELL_W) * np.sin(q[3]), 0],
            [0, 1],
        ]
    )

    # (b) Compute the state covariance matrix and state update
    P_new = F @ P @ F.T + G @ Q @ G.T
    P_new = 0.5 * (P_new + P_new.T)
    q_new = q + T * vehicle.f(q, u)

    # Compute the a posteriori estimate
    # (a) Compute the Jacobian matrices (i.e., linearize about current estimate)
    H = np.zeros((N_FEATURES + 1, 4))
    for j in range(0, N_FEATURES):
        H[j, :] = np.array(
            [
                -(f_map[0, j] - q[0]) / range_sensor(q, 0, f_map)[j],
                -(f_map[1, j] - q[1]) / range_sensor(q, 0, f_map)[j],
                0,
                0,
            ]
        )
    # Add a measurement for the steering angle
    H[N_FEATURES, :] = np.array([0, 0, 0, 1])

    # Check the observability of this system
    observability_matrix = H
    for j in range(1, 4):
        observability_matrix = np.concatenate(
            (observability_matrix, H @ np.linalg.matrix_power(F, j)), axis=0
        )
    if np.linalg.matrix_rank(observability_matrix) < 4:
        raise ValueError("System is not observable!")

    # (b) Compute the Kalman gain
    K = P_new @ H.T @ inv(H @ P_new @ H.T + R)

    # (c) Compute the state update
    z_hat = np.zeros(N_FEATURES + 1)
    z_hat[0:N_FEATURES] = range_sensor(q_new, 0, f_map)
    z_hat[N_FEATURES] = q_new[3]
    q_new = q_new + K @ (z - z_hat)

    # (d) Compute the covariance update
    P_new = (np.eye(4) - K @ H) @ P_new @ (np.eye(4) - K @ H).T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    # Return the estimated state
    return q_new, P_new


# %%
# RUN SIMULATION

# Initialize arrays that will be populated with our inputs and states
x = np.zeros((4, N))
u = np.zeros((2, N))
x_hat = np.zeros((4, N))
P_hat = np.zeros((4, 4, N))

# Set the initial pose [m, m, rad, rad], velocities [m/s, rad/s]
x[0, 0] = -2.0
x[1, 0] = 3.0
x[2, 0] = np.pi/8.0
x[3, 0] = 0.0
u[0, 0] = 2.0
u[1, 0] = 0

# Initialize the state estimate
x_hat[:, 0] = np.array([0.0, 0.0, 0.0, 0.0])
P_hat[:, :, 0] = np.diag([5.0**2, 5.0**2, (np.pi/4)**2, 0.05**2])

# Just drive around and try to localize!
for k in range(1, N):
    # Simulate the robot's motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)
    # Measure the actual range to each feature
    z = np.zeros(N_FEATURES + 1)
    z[0:N_FEATURES] = range_sensor(x[:, k], SIGMA_W1, f_map)
    z[N_FEATURES] = x[3, k] + SIGMA_W2 * np.random.randn()

    # Use the range measurements to estimate the robot's state
    x_hat[:, k], P_hat[:, :, k] = fws_ekf(
        x_hat[:, k - 1], P_hat[:, :, k - 1], u[:, k - 1], z, Q, R, f_map
    )
    # Choose some new inputs
    # Choose some new inputs
    u[0, k] = 2.0
    u[1, k] = -0.1 * np.sin(1 * t[k])

# %%
# MAKE SOME PLOTS


# Function to wrap angles to [-pi, pi]
def wrap_to_pi(angle):
    """Wrap angles to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Find the scaling factor for plotting covariance bounds
ALPHA = 0.01
s1 = chi2.isf(ALPHA, 1)
s2 = chi2.isf(ALPHA, 2)

# Plot the position of the vehicle in the plane
fig1 = plt.figure(1)
plt.plot(f_map[0, :], f_map[1, :], "C4*", label="Feature")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.plot(x_hat[0, :], x_hat[1, :], "C1--", label="Estimated")
plt.axis("equal")
X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = vehicle.draw(x[:, 0])
plt.fill(X_BL, Y_BL, "k")
plt.fill(X_BR, Y_BR, "k")
plt.fill(X_FR, Y_FR, "k")
plt.fill(X_FL, Y_FL, "k")
plt.fill(X_BD, Y_BD, "C2", alpha=0.5, label="Start")
X_BL, Y_BL, X_BR, Y_BR, X_FL, Y_FL, X_FR, Y_FR, X_BD, Y_BD = vehicle.draw(x[:, N - 1])
plt.fill(X_BL, Y_BL, "k")
plt.fill(X_BR, Y_BR, "k")
plt.fill(X_FR, Y_FR, "k")
plt.fill(X_FL, Y_FL, "k")
plt.fill(X_BD, Y_BD, "C3", alpha=0.5, label="End")
plt.xlabel(r"$x$ [m]")
plt.ylabel(r"$y$ [m]")
plt.legend()

# Plot the states as a function of time
fig2 = plt.figure(2)
fig2.set_figheight(6.4)
ax2a = plt.subplot(411)
plt.plot(t, x[0, :], "C0", label="Actual")
plt.plot(t, x_hat[0, :], "C1--", label="Estimated")
plt.grid(color="0.95")
plt.ylabel(r"$x$ [m]")
plt.setp(ax2a, xticklabels=[])
plt.legend()
ax2b = plt.subplot(412)
plt.plot(t, x[1, :], "C0", label="Actual")
plt.plot(t, x_hat[1, :], "C1--", label="Estimated")
plt.grid(color="0.95")
plt.ylabel(r"$y$ [m]")
plt.setp(ax2b, xticklabels=[])
ax2c = plt.subplot(413)
plt.plot(t, wrap_to_pi(x[2, :]) * 180.0 / np.pi, "C0", label="Actual")
plt.plot(t, wrap_to_pi(x_hat[2, :]) * 180.0 / np.pi, "C1--", label="Estimated")
plt.ylabel(r"$\theta$ [deg]")
plt.grid(color="0.95")
plt.setp(ax2c, xticklabels=[])
ax2d = plt.subplot(414)
plt.plot(t, wrap_to_pi(x[3, :]) * 180.0 / np.pi, "C0", label="Actual")
plt.plot(t, wrap_to_pi(x_hat[3, :]) * 180.0 / np.pi, "C1--", label="Estimated")
plt.ylabel(r"$\phi$ [deg]")
plt.grid(color="0.95")
plt.xlabel(r"$t$ [s]")

# Plot the estimator errors as a function of time
sigma = np.zeros((4, N))
fig3 = plt.figure(3)
fig3.set_figheight(6.4)
ax3a = plt.subplot(411)
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(
    t,
    -sigma[0, :],
    sigma[0, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, x[0, :] - x_hat[0, :], "C0", label="Error")
plt.grid(color="0.95")
plt.ylabel(r"$e_x$ [m]")
plt.setp(ax2a, xticklabels=[])
plt.legend()
ax3b = plt.subplot(412)
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(
    t,
    -sigma[1, :],
    sigma[1, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, x[1, :] - x_hat[1, :], "C0", label="Error")
plt.grid(color="0.95")
plt.ylabel(r"$e_y$ [m]")
plt.setp(ax2b, xticklabels=[])
ax3c = plt.subplot(413)
sigma[2, :] = np.sqrt(s1 * P_hat[2, 2, :])
plt.fill_between(
    t,
    -sigma[2, :] * 180.0 / np.pi,
    sigma[2, :] * 180.0 / np.pi,
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, (x[2, :] - x_hat[2, :]) * 180.0 / np.pi, "C0", label="Error")
plt.ylabel(r"$e_\theta$ [deg]")
plt.grid(color="0.95")
plt.setp(ax2c, xticklabels=[])
ax3d = plt.subplot(414)
sigma[3, :] = np.sqrt(s1 * P_hat[3, 3, :])
plt.fill_between(
    t,
    -sigma[3, :] * 180.0 / np.pi,
    sigma[3, :] * 180.0 / np.pi,
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, (x[3, :] - x_hat[3, :]) * 180 / np.pi, "C0", label="Error")
plt.ylabel(r"$e_\phi$ [deg]")
plt.grid(color="0.95")
plt.xlabel(r"$t$ [s]")

# Show all the plots to the screen
plt.show()

# %%
