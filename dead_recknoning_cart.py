"""
Example dead_reckoning_cart.py
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Set the simulation time [s] and the sample period [s]
SIM_TIME = 5.0
T = 0.1

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# %%
# DEFINE THE ROBOT MODEL

# Mobile robot model
F = np.array([[1, T], [0, 1]])
G = np.array([0.5 * T**2, T])

# Input noise covariance matrix
Q = 1.0**2

# %%
# SET UP INITIAL CONDITIONS

# Set up arrays for a two-dimensional state
x_hat = np.zeros((2, N))
P_hat = np.zeros((2, 2, N))

# Robot state (actual)
x = np.zeros([2, N])

# Initial estimate of the robot's location
x_hat[:, 0] = x[:, 0]
# Initial covariance matrix
P_hat[:, :, 0] = 0.0 * np.eye(2)

# Initial inputs
u = np.zeros(1)
u_measured = np.zeros(1)

# %%
# RUN THE SIMULATION

for k in range(1, N):
    # Simulate the actual vehicle
    x[:, k] = F @ x[:, k - 1] + G * u

    # Run the estimator (dead reckoning)
    x_hat[:, k] = F @ x_hat[:, k - 1] + G * u_measured
    P_hat[:, :, k] = F @ P_hat[:, :, k - 1] @ F.T + G * Q @ G.T
    # Help the covariance matrix stay symmetric
    P_hat[:, :, k] = 0.5 * (P_hat[:, :, k] + P_hat[:, :, k].T)

    # Control input (actual)
    u = 10 * np.sin(2 * t[k])

    # Measured input (with zero-mean Gaussian noise)
    u_measured = u + np.sqrt(Q) * np.random.randn()

# %%
# PLOT THE RESULTS

# Change some plot settings (optional)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Find the scaling factor for plotting covariance bounds
ALPHA = 0.01
s1 = chi2.isf(ALPHA, 1)
s2 = chi2.isf(ALPHA, 2)

# Plot the state and the estimate
sigma = np.zeros((2, N))
plt.figure()
ax1 = plt.subplot(211)
sigma[0, :] = np.sqrt(s1 * P_hat[0, 0, :])
plt.fill_between(
    t,
    x_hat[0, :] - sigma[0, :],
    x_hat[0, :] + sigma[0, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, x[0, :], "C0", label="Actual")
plt.plot(t, x_hat[0, :], "C1", label="Estimate")
plt.ylabel(r"$x_1$ [m]")
plt.setp(ax1, xticklabels=[])
plt.grid(color="0.95")
plt.legend()
ax2 = plt.subplot(212)
sigma[1, :] = np.sqrt(s1 * P_hat[1, 1, :])
plt.fill_between(
    t,
    x_hat[1, :] - sigma[1, :],
    x_hat[1, :] + sigma[1, :],
    color="C0",
    alpha=0.2,
    label=str(100 * (1 - ALPHA)) + r" \% Confidence",
)
plt.plot(t, x[1, :], "C0", label="Actual")
plt.plot(t, x_hat[1, :], "C1", label="Estimate")
plt.ylabel(r"$x_2$ [m]")
plt.grid(color="0.95")
plt.xlabel(r"$t$ [s]")
plt.show()

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
plt.plot(t, x[0, :] - x_hat[0, :], "C0", label="Estimator Error")
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
plt.plot(t, x[1, :] - x_hat[1, :], "C0", label="Estimator Error")
plt.ylabel(r"$e_2$ [m]")
plt.grid(color="0.95")
plt.xlabel(r"$t$ [s]")
plt.show()
