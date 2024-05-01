from envs.resources.settings import *
from envs.resources.functions import *
import math

# The pod class contains the information for each pod:
#   -id is an index to be matched with actions and states.
#   - x and y are the current position of the pod
#   - t_x and t_y are the point that the pod is currently traveling towards
#   - target is the index of checkpoins that the pod should hit next
#   - color is the color to be rendered as; the default is red. 
class Pod():
    def __init__(self, id, x, y, t_x, t_y, target = 1, color = RED):
        self.id = id
        self.color = tuple(color)
        self.x = x
        self.y = y
        self.target = target
        self.angle = math.atan2((t_y - y),(t_x - x))

        self.thrust = 0
        self.x_speed = 0
        self.y_speed = 0
        self.x_acceleration = 0
        self.y_acceleration = 0

        self.checked = 0

    # Returns the current position
    def get_pos(self):
        return (int(self.x), int(self.y))
    
    # This function updates the current postion and angle of the pod, based on the given (x, y, thrust) action.
    def update(self, x, y, thrust):
        
        self.thrust = 661 if thrust == 101 else thrust
        theta = normalize_angle(check_angle(self.x, self.y, x, y))
        
        if theta != self.angle:
            self.angle = update_angle(theta, self.angle)
            
        x_power = round(math.cos(self.angle) * self.thrust,0)
        y_power = round(math.sin(self.angle) * self.thrust,0)
        self.x_acceleration = math.floor(self.x_speed * 0.85)
        self.y_acceleration = math.floor(self.y_speed * 0.85)
        self.x_speed = self.x_acceleration + x_power
        self.y_speed = self.y_acceleration + y_power

        self.x += (self.x_speed)
        self.y += (self.y_speed)
        
        
