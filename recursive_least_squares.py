"""
Example recursive_least_squares.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 2.0
T = 0.1

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# FUNCTION DEFINITIONS


def gps_sensor(x, R, H):
    """Simulated GPS sensor."""
    y = H @ x + R @ np.random.randn(np.size(x))
    return y


def rls_estimator(x, P, y, R, H):
    n = np.size(x)
    K = P @ H.T @ inv(H @ P @ H.T + R)
    x_next = x + K @ (y - H @ x)
    P_next = (np.eye(n) - K @ H) @ P @ (np.eye(n) - K @ H).T + K @ R @ K.T
    return x_next, P_next


# %%
# SET UP INITIAL CONDITIONS

# Set up arrays for a two-dimensional GPS measurements problem
x_hat = np.zeros((2, N))
P_hat = np.zeros((2, 2, N))
y = np.zeros((2, N))

# Robot location (actual)
x = np.array([2.0, 4.0])
# Initial estimate of the robot's location
x_hat[:, 0] = np.array([0.0, 0.0])
# Initial covariance matrix
P_hat[:, :, 0] = 10.0 * np.eye(2)
# Measurement noise covariance matrix
SIGMA = 1.0
R = SIGMA**2 * np.eye(2)
# Measurement matrix (i.e., GPS coordinates)
H = np.eye(2)

# %%
# RUN THE SIMULATION

for k in range(1, N):
    # Simulate the GPS sensor
    y[:, k] = gps_sensor(x, R, H)
    # Update the estimate using the recursive least squares estimator
    x_hat[:, k], P_hat[:, :, k] = rls_estimator(
        x_hat[:, k - 1], P_hat[:, :, k - 1], y[:, k], R, H
    )

# %%
# PLOT THE RESULTS

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the true and estimated robot location
plt.figure()
plt.plot(x[0], x[1], "C4o", label="True")
plt.plot(x_hat[0, :], x_hat[1, :], "C0-", label="Estimated")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.legend()
plt.grid(color="0.95")
plt.axis("equal")

# Find the scaling factor for plotting covariance bounds
ALPHA = 0.01
s1 = chi2.isf(ALPHA, 1)
s2 = chi2.isf(ALPHA, 2)

# Plot the error in the estimate
sigma = np.zeros((2, N))
plt.figure()
ax1 = plt.subplot(211)
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(
    t,
    -sigma[0, :],
    sigma[0, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, x[0] - x_hat[0, :], "C0", label="Estimator Error")
plt.plot(t, x[0] - y[0, :], "C1", label="Raw Measurement Error")
plt.ylabel(r"$e_1$ [m]")
plt.setp(ax1, xticklabels=[])
plt.grid(color="0.95")
plt.legend()
ax2 = plt.subplot(212)
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(
    t,
    -sigma[1, :],
    sigma[1, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, x[1] - x_hat[1, :], "C0", label="Estimator Error")
plt.plot(t, x[1] - y[1, :], "C1", label="Raw Measurement Error")
plt.ylabel(r"$e_2$ [m]")
plt.grid(color="0.95")
plt.xlabel(r"$t$ [s]")
plt.show()

# %%
