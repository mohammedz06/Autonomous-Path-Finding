"""
Example control_approx_linearization.py (Modified for QVEX APSC 103 Software Project)
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

# %%
# SIMULATION SETUP

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal
from math import pi
from mobotpy.models import DiffDrive
from mobotpy.integration import rk_four
from apsc_103.path_model import *

# %%
# COMPUTE THE PATH
"""
Insert your path planning code in this cell. You can write it in another
file and call it from here, or you can write it all in here if you prefer. 
Your algorithm should output a path array with the line and arc segments
that it outputs as the path of best fit. The path array below is just an
example and can be replaced with your code's output. 
You do not really have to worry about the code in the other cells, unless you 
have to change a parameter for whatever reason. 
"""
goal_points = [
    (10, 5),
    (6, 2),
]


show_walls = True 
enable_obstacles = True # Change based on whether you'd like to enable obstacles
# Midpoint(s) of 4'x4' pallet(s):
obstacles = [
    Obstacle(4, 5),
    Obstacle(20, 20),
]

path = [
    LineSegment(1, 1, 5, 1),
    ArcSegment(6, 4, 3, -pi/2, 0),
]

# %%
# COMPUTE THE REFERENCE TRAJECTORY

# Set the simulation time [s] and the sample period [s]
SIM_TIME = sum(segment.duration() for segment in path)
T = 0.04

# Create an array of time values [s]
t = np.arange(0.0, SIM_TIME, T)
N = np.size(t)

# # Radius of the circle [m]
# R = 10

# # Angular rate [rad/s] at which to traverse the circle
# OMEGA = 0.1

# Pre-compute the desired trajectory
x_d = np.zeros((3, N))
u_d = np.zeros((2, N))
segment_time = 0.0
# t1 = 0

# Loop to iterate over segments of specified line and arc segment: 
max_k = -1
for segment in path: 
    segment_duration = segment.duration()
    for k in range (int(segment_time/T), int((segment_time + segment_duration)/T)):
        t1 = (k - (segment_time/T))*T
        x_d[0, k], x_d[1, k] = segment.position_at_point(t1) # x and y coords
        x_d[2, k] = segment.orientation_at_point(t1) # Orientation
        u_d[0, k] = VELOCITY # Forward velocity
        u_d[1, k] = segment.angular_velocity() # Angular velocity
        max_k = k
    segment_time += segment_duration
if max_k < N - 1:
    N = max_k + 1
    t = t[:N]
    x_d = x_d[:, :N]
    u_d = u_d[:, :N]


# %%
# VEHICLE SETUP

# Set the track length of the vehicle [m]
ELL = 1.0

# Create a vehicle object of type DiffDrive
vehicle = DiffDrive(ELL)

# %%
# SIMULATE THE CLOSED-LOOP SYSTEM

# Initial conditions
x_init = np.zeros(3)
x_init[0] = 1.0
x_init[1] = 1.0
x_init[2] = 0.0

# Setup some arrays
x = np.zeros((3, N))
u = np.zeros((2, N))
x[:, 0] = x_init

for k in range(1, N):

    # Simulate the differential drive vehicle motion
    x[:, k] = rk_four(vehicle.f, x[:, k - 1], u[:, k - 1], T)

    # Compute the approximate linearization
    A = np.array(
        [
            [0, 0, -u_d[0, k - 1] * np.sin(x_d[2, k - 1])],
            [0, 0, u_d[0, k - 1] * np.cos(x_d[2, k - 1])],
            [0, 0, 0],
        ]
    )
    B = np.array([[np.cos(x_d[2, k - 1]), 0], [np.sin(x_d[2, k - 1]), 0], [0, 1]])

    try:
        # Compute the gain matrix to place poles of (A - BK) at p
        p = np.array([-1.0, -2.0, -0.5])
        K = signal.place_poles(A, B, p)
    except ValueError as err:
        print(f"k = {k}")
        print(f"A = {A}")
        print(f"B = {B}")
        raise err

    # Compute the controls (v, omega) and convert to wheel speeds (v_L, v_R)
    u_unicycle = -K.gain_matrix @ (x[:, k - 1] - x_d[:, k - 1]) + u_d[:, k]
    u[:, k] = vehicle.uni2diff(u_unicycle)
          

# %%
# MAKE PLOTS

# Change some plot settings (optional)
# plt.rc("text", usetex=True)
# plt.rc("text.latex", preamble=r"\usepackage{cmbright,amsmath,bm}")
plt.rc("savefig", format="pdf")
plt.rc("savefig", bbox="tight")

# Plot the states as a function of time
fig1 = plt.figure(1)
fig1.set_figheight(6.4)
ax1a = plt.subplot(411)
plt.plot(t, x_d[0, :], "C1--") 
plt.plot(t, x[0, :], "C0")
plt.grid(color="0.95")
plt.ylabel("x [feet]")
plt.setp(ax1a, xticklabels=[])
plt.legend(["Desired", "Actual"])
ax1b = plt.subplot(412)
plt.plot(t, x_d[1, :], "C1--")
plt.plot(t, x[1, :], "C0")
plt.grid(color="0.95")
plt.ylabel("y [feet]")
plt.setp(ax1b, xticklabels=[])
ax1c = plt.subplot(413)
plt.plot(t, x_d[2, :] * 180.0 / np.pi, "C1--")
plt.plot(t, x[2, :] * 180.0 / np.pi, "C0")
plt.grid(color="0.95")
plt.ylabel("θ [deg]")
plt.setp(ax1c, xticklabels=[])
ax1d = plt.subplot(414)
plt.step(t, u[0, :], "C2", where="post", label="$v_L$")
plt.step(t, u[1, :], "C3", where="post", label="$v_R$")
plt.grid(color="0.95")
plt.ylabel("u [feet/s]")
plt.xlabel("t [s]")
plt.legend()

# Save the plot
# plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig1.pdf")

# Plot the position of the vehicle in the plane
fig2 = plt.figure(2)
ax = fig2.add_subplot(1, 1, 1)
plt.plot(x_d[0, :], x_d[1, :], "C1--", label="Desired")
plt.plot(x[0, :], x[1, :], "C0", label="Actual")
plt.axis("equal")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(x[0, 0], x[1, 0], x[2, 0])
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C2", alpha=0.5, label="Start")
X_L, Y_L, X_R, Y_R, X_B, Y_B, X_C, Y_C = vehicle.draw(
    x[0, N - 1], x[1, N - 1], x[2, N - 1]
)
plt.fill(X_L, Y_L, "k")
plt.fill(X_R, Y_R, "k")
plt.fill(X_C, Y_C, "k")
plt.fill(X_B, Y_B, "C3", alpha=0.5, label="End")
plt.xlabel("x [feet]")
plt.ylabel("y [feet]")
plt.legend()

# Make border of warehouse
if show_walls == True:
    warehouse_border = patches.Rectangle((0, 0), 50, 25, fill=False, linewidth=1)
    ax.add_patch(warehouse_border)

# Show points 
for point in goal_points:
    point = patches.Circle(point, radius=0.25, color='r')
    ax.add_patch(point)

# Plot the position of the obstacle 
# Obstacle is a 4'x4' square, and the point given refers to the midpoint of the obstacle
if enable_obstacles == True:
    for obstacle in obstacles:
        rect = patches.Rectangle([(obstacle.x-2), (obstacle.y-2)], 4, 4)
        ax.add_patch(rect)

# Save the plot
# plt.savefig("../agv-book/figs/ch4/control_approx_linearization_fig2.pdf")

# Show the plots to the screen
# plt.show()

# %%
# OBSTACLE AND POINT HITS
# Find if obstacle has been hit by robot
def circle_intersects_rectangle(circle_center, rect_center, rect_size):
    # Circle and rectangle parameters
    x_c, y_c = circle_center
    x, y = rect_center
    half_size = rect_size / 2
    x_min, x_max = x - half_size, x + half_size
    y_min, y_max = y - half_size, y + half_size

    # Is circle's center is inside the rectangle
    if x_min <= x_c <= x_max and y_min <= y_c <= y_max:
        return True  

    # distance to rectangle edges
    closest_x = np.clip(x_c, x_min, x_max)
    closest_y = np.clip(y_c, y_min, y_max)

    # distance from the circle's center to this closest point
    distance = np.sqrt((closest_x - x_c)**2 + (closest_y - y_c)**2)
    if distance <= ELL:
        return True 

    # Check if any rectangle corner is inside the circle
    corners = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
    for corner_x, corner_y in corners:
        distance_to_corner = np.sqrt((corner_x - x_c)**2 + (corner_y - y_c)**2)
        if distance_to_corner <= ELL:
            return True  # Circle intersects the rectangle at a corner

    return False  # No intersection found

def check_robot_path(x_coords, y_coords, rect_center, rect_size):
    hit = False
    for x_c, y_c in zip(x_coords, y_coords):
        if circle_intersects_rectangle((x_c, y_c), rect_center, rect_size):
            hit = True
    return hit

x_coords = x[0, :]
y_coords = x[1, :]
if enable_obstacles == True: 
    hit = False
    for obstacle in obstacles:
        hit = check_robot_path(x_coords, y_coords, (obstacle.x, obstacle.y), 4)
        if hit == True:
            print("The robot hit an obstacle")

# Check if wall has been hit
wall_hit = False
for x1, y1 in zip(x_coords, y_coords):
    if (x1 - ELL) < 0 or (x1 + ELL) > 50 or (y1 - ELL) < 0 or (y1+ELL) > 25:
        wall_hit = True       
if wall_hit == True: 
    print("Robot hit a wall of the warehouse")

hit_points = set()
# Check if robot has made it to a point:
for x_c, y_c in zip(x_coords, y_coords):
    for x_goal, y_goal in goal_points:
        # Compute dist from the robot center to the target point
        distance = np.sqrt((x_c - x_goal)**2 + (y_c - y_goal)**2)

        # Check if dist is less than or equal to the radius
        if distance <= ELL:
            hit_points.add((x_goal, y_goal))
if len(hit_points) > 0:
    print(f"The robot successfully got to the point(s): {hit_points}")

# %%
# MAKE AN ANIMATION

# Create the animation
obstacle_coords = []
if enable_obstacles == True:
    for obstacle in obstacles:
        obstacle_coords.append(((obstacle.x-2), (obstacle.y-2)))
    ani = ani = vehicle.animate_trajectory(x, x_d, T, True, "animation.gif", goal_points, True, obstacle_coords)
else: 
    ani = vehicle.animate_trajectory(x, x_d, T, True, "animation.gif", goal_points, False)

# Create and save the animation
# ani = vehicle.animate_trajectory(
#     x, x_d, T, True, "control_approx_linearization.gif"
# )


# Show all the plots to the screen
plt.show()

# Show animation in HTML output if you are using IPython or Jupyter notebooks
# from IPython.display import display

# plt.rc("animation", html="jshtml")
# display(ani)
# plt.close()

# %%
