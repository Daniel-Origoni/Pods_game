from envs.resources.settings import *
import math

# functions used by the environment

# Given two points (x1, y1) and (x2, y2), it will calculate the angle between them with respect to the x-axis
def check_angle(x1, y1, x2, y2):
    return math.atan2((y2 - y1),(x2 - x1))

# Given two points (x1, y1) and (x2, y2), it will calculate the distance between them
def checkDistance(x1, y1, x2, y2):
    return math.floor(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))

# It returns a list of the current location for all pods
def get_info(pods):
    info = ()
    for pod in pods:
        info += pod.get_pos()
    return info

# It returns a list of the location of the all player-pods, and the location of the next checkpoint for each
def get_obs(pods, num_players, checkpoints):
    agent_location = [[pods[i].get_pos() for i in range(num_players)]]
    target_location = [[checkpoints[pods[i].target] for i in range(num_players)]]
    obs = [*agent_location, *target_location]
    return obs

# Given an angle, it will retrun the points of a triangle to draw the pods, rotated theta degrees around the origin.
# The points can then be translated to the right location in the map.
def get_triangle(theta):
    p1 = (36.22, 0)
    p2 = (-28.38, 22.5)
    p3 = (-28.38, -22.5)

    rp1x = p1[0] * math.cos(theta) - p1[1] * math.sin(theta)
    rp1y = p1[0] * math.sin(theta) + p1[1] * math.cos(theta)                 
    rp2x = p2[0] * math.cos(theta) - p2[1] * math.sin(theta)
    rp2y = p2[0] * math.sin(theta) + p2[1] * math.cos(theta)                        
    rp3x = p3[0] * math.cos(theta) - p3[1] * math.sin(theta)                         
    rp3y = p3[0] * math.sin(theta) + p3[1] * math.cos(theta)
    rp1 = ( rp1x, rp1y )
    rp2 = ( rp2x, rp2y )
    rp3 = ( rp3x, rp3y )

    return (rp1, rp2, rp3)

# The function returns an angle between 0 and 2pi (360 degrees) for an angle that is out of range
def normalize_angle(angle):
    normal = 2 * math.pi
    return (angle + normal) % normal

# This function limits the rotation of the pod to a specific rotation speed.
# Given the current angle of the pod and a desired angle, it will rotate the triangle upto the rotation speed.
# If the resultin angle is outisde the range 0 to 2pi (360 degrees), it will return an angle within that range.
def update_angle(theta, angle):
    if angle < math.pi and theta > angle:
        if theta < angle + math.pi:
            if (angle + ROTATION_SPEED) > theta:
                angle = theta
            else:
                angle += ROTATION_SPEED
        else:
            if (angle - ROTATION_SPEED) > 0:
                angle -= ROTATION_SPEED
            elif normalize_angle(angle - ROTATION_SPEED) < theta:
                angle = theta
            else:
                angle = normalize_angle(angle - ROTATION_SPEED)

    elif theta < angle:
        if theta > angle - math.pi:
            if (angle - ROTATION_SPEED) < theta:
                angle = theta
            else:
                angle -= ROTATION_SPEED
        else:
            if (angle + ROTATION_SPEED) < (math.pi * 2):
                angle += ROTATION_SPEED
            elif normalize_angle(angle + ROTATION_SPEED) > theta:
                angle = theta
            else:
                angle = normalize_angle(angle + ROTATION_SPEED)

    else: 
        if (angle + ROTATION_SPEED) > theta:
                angle = theta
        else:
                angle += ROTATION_SPEED

    return angle

