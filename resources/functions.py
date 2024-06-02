from resources.settings import *
import numpy as np
import math

def normalize_state(states):
    states = np.reshape(list(zip(*states)), (len(states[0]),4)).tolist()
    for i in range(len(states)):
        states[i].insert(len(states[i]), ((states[i][0] - states[i][2])**2 + (states[i][1] - states[i][3])**2)**0.5)
        states[i] = [i / j for i, j in zip(states[i], [WIDTH, HEIGHT, WIDTH, HEIGHT, (WIDTH **2 + HEIGHT**2)**0.5])]
        states[i] = np.reshape(states[i], (5,1))
    return states

def normalize_action(actions):
    normalized_actions = []
    for action in actions:
        action += np.array([1, 1, 1])
        action = action / [2, 2, 2]
        action *= np.array([WIDTH, HEIGHT, 101])
        action[2] = math.floor(action[2])
        normalized_actions.append(action)
    return normalized_actions

def mse(y_target, y_pred):
    return np.mean(np.power(y_target - y_pred, 2))

def mse_prime(y_target, y_pred):
    return 2 * (y_pred - y_target) / np.size(y_target)