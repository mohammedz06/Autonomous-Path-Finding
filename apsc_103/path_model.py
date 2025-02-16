"""
This file declares the classes LineSegment and ArcSegment, 
which are used to model the path and generate the trajectory. 
You should initialize the start and end x & y values when 
declaring a LineSegment, and the center coordinates, radius, 
and start and end angles when declaring an ArcSegment.
"""
import math

# Velocity is constant
VELOCITY = 1.0 # [feet/s]

class LineSegment:
    def __init__(self, start_x, start_y, end_x, end_y) -> None:
        # Set initial conditions:
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y
        self.length = math.sqrt((self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)
    def duration(self) -> float:
        return self.length / VELOCITY
    def position_at_point(self, t):
        # Position of robot at time t: 
        dist = VELOCITY * t
        ratio = dist / self.length
        x = self.start_x + ratio * (self.end_x - self.start_x)
        y = self.start_y + ratio * (self.end_y - self.start_y)
        return x, y
    def orientation_at_point(self, t):
        return math.atan2(self.end_y - self.start_y, self.end_x - self.start_x)
    def angular_velocity(self):
        return 0.0


class ArcSegment:
    def __init__(self, center_x, center_y, radius, start_angle, end_angle) -> None:
        # Set initial conditions: 
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.length = self.radius * abs(self.end_angle - self.start_angle)
    def duration(self) -> float:
        return self.length / VELOCITY
    def position_at_point(self, t):
        # Find robot position at time t: 
        distance = t * VELOCITY # [feet]
        if self.end_angle < self.start_angle:
            distance = -distance
        angle = self.start_angle + distance / self.radius # [rad]
        x = self.center_x + self.radius * math.cos(angle)
        y = self.center_y + self.radius * math.sin(angle)
        return x, y
    def orientation_at_point(self, t):
        if self.end_angle > self.start_angle: # counterclockwise
            return self.start_angle + t * VELOCITY / self.radius + 0.5 * math.pi
        else: # clockwise
            return self.start_angle - t * VELOCITY / self.radius + 0.5 * math.pi
    def angular_velocity(self):
        return VELOCITY / self.radius
    
class Obstacle: 
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        
