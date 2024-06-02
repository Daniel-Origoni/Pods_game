from agents.AI import Agent
from envs.pod_racing import RaceTrackEnv
from resources import *

import numpy as np

# Set some training settings
buffer_size = 500
batch_size = 100
epochs = 100
num_agents = 1

# Set the playing environment to render and a trainign environment without rendering
training_env = RaceTrackEnv(render_mode="rgb_array", num_players = num_agents, num_bots = 0)
playing_env = RaceTrackEnv(render_mode="human", render_fps = 20, num_players = num_agents, num_laps = 5)

# Populate a list of agents
agents = []
for _ in range(num_agents):
    agents.append(Agent(buffer_size = buffer_size))
    

# The main loop shows what the agent/s do for 4 laps
# Then trains the agent/s for as many epochs
# Then plays shows the user another 4 laps after training. 

done = False
states, info = playing_env.reset()
states = normalize_state(states)
while not done:
        prev_state = states
        actions = []
        
        for i, agent in enumerate(agents):
            
            action = np.resize(agent.forward(states[i]), (3,))
            actions.append(action)

        states, rewards, done, info = playing_env.step(normalize_action(actions))
        states = normalize_state(states)



for e in range(epochs):
    eps = EPSILON
    done = False
    states, info = training_env.reset()
    states = normalize_state(states)
    steps = 0
    indices = []

    print("Epoch: " + str(e))

    while not done:
        prev_state = states
        actions = []
                
        # Given a list of states, populate the list of actions, with the action for each agent. 
        for i, agent in enumerate(agents):
            if np.random.random() < eps:
                action = np.random.uniform(low=-1, high= 1, size=(3,))
            else:
                action = np.resize(agent.forward(states[i]), (3,))
            actions.append(action)

        # Given the list of actions, return a list of states and rewards.
        states, rewards, done, info = training_env.step(normalize_action(actions))
        states = normalize_state(states)

        for i, agent in enumerate(agents):
            agent.learn(states[i], actions[i], rewards[i], states[i], batch_size)

        steps += 1
        if steps >= 500:
            done = True
        
        if eps > 0:
            eps -= 0.005

    
done = False
states, info = playing_env.reset()
states = normalize_state(states)
while not done:
        prev_state = states
        actions = []
        
        for i, agent in enumerate(agents):
            
            action = np.resize(agent.forward(states[i]), (3,))
            actions.append(action)

        states, rewards, done, info = playing_env.step(normalize_action(actions))
        states = normalize_state(states)