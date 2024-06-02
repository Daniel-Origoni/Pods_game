import numpy as np
import copy
from agents.resources import *

# The Agent class consists of 4 Brains: an actor, a critic, a target_actor and a target_critic.
#   The targe_networks are initialized as identical copies of the actor and ciritc networks,
#   but are updated one step behind. 
#
#   The shape and structure of each network is defined in the settings.py file, along with
#   the learning rate, and epsilon value for an epsilon-greedy strategy. 
#
# During the learning step, the agent will store experiences into the memory buffer.
#   Each experience includes the current state that the agent made a decision on,
#   the action that the eagent chose to take,
#   the reward given by the environment for that action, 
#   and the next state returned by the environment. 
#
#   Once the memory buffer is full, at each step the agent will sample a given number of experiences.
#   For each sample in the batch, the agent will:
#       generate a new action, using the target_actor, given the next_state,
#       calculate a V_value using the target_critic network, given the next_state-next_action pair,
#       calculate the target_q value as the sum of the reward from the previous action and the discounted V_value
#       calculate the predicted_q value using the critic network, given the previous state-action pair. 
#       calculate the critic loss as the square difference between the predicted_q and the target_q
#       calculate the gradient of the ciritic, as the derivative of the loss with respect to the predicted_q
#       calculate the gradient of the actor network as the output gradient from the critic network
#           from the input nodes that correspond to the action

class Agent:
    def __init__(self, buffer_size = 1000):                
        self.actor = Brain(STATE_DIM, ACTION_DIM, HIDDEN_DIM, ACTOR_NETWORK, ACTOR_LEARNING_RATE)
        self.critic = Brain(STATE_DIM + ACTION_DIM, 1, HIDDEN_DIM, CRITIC_NETWORK, CRITIC_LEARNING_RATE)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.memory = ReplayBuffer(buffer_size)

        self.C_loss = []
        self.A_loss = []
        self.rewards = 0

    def forward(self, state):
        return self.actor.forward(state)
        

    def learn(self, state, action, reward, next_state, batch_size=250):
       
        self.memory.add_experience(state, action, reward, next_state)
        self.rewards += reward


        if len(self.memory.buffer) >= self.memory.buffer_size:
            states, actions, rewards, next_states = self.memory.sample_batch(batch_size)
            critic_loss = []
            critic_gradient = np.zeros([1,1])
            actor_loss = []
            
            for i in range(batch_size):
                next_action = self.target_actor.forward(next_states[i])

                target_critic_input = np.concatenate((next_states[i], next_action))
                target_critic_input = np.reshape(target_critic_input, (8,1))
                V_value = self.target_critic.forward(target_critic_input) *2

                target_q = rewards[i] + DISCOUNT_FACTOR * V_value

                critic_input = np.concatenate((*states[i], actions[i]))
                critic_input = np.reshape(critic_input, (8,1))
                predicted_q = self.critic.forward(critic_input) *2    

                critic_gradient += 2*(predicted_q - target_q)

                critic_loss.insert(len(critic_loss),(target_q - predicted_q)**2)
                actor_loss.insert(len(actor_loss), -predicted_q)
                    

            
            critic_loss = sum(critic_loss) * (1/batch_size)
            critic_gradient *= (1/batch_size)
            actor_loss = sum(actor_loss) * (1/batch_size) 

            self.C_loss.insert(len(self.C_loss), critic_loss[0])
            self.A_loss.insert(len(self.A_loss), actor_loss[0])

            self.target_actor = copy.deepcopy(self.actor)
            self.target_critic = copy.deepcopy(self.critic)
            actor_gradient = self.critic.backward(critic_gradient)
            self.actor.backward(actor_gradient[5:])
        
# The Brain class builds a network based on the structure passed in the thrid argument.
#   The structure the network consists of layers, each layer defined in the layers.py file
#   Currently it only supports Dense (fully-connected), Tanh, and Linear layers.
#       Starting from the end of the network structure, the first time that a Dense layer is added,
#           the size of the output will be equal to the output_dim.
#       When the index reaches the begining of the nextwork structure, if the layers is Dense,
#           the size of the input will be euqual to the input_dim. 
#
#   The forward and backward method iterate through the layers (forwards or backwards),
#       calling the corresponding mthod from each layer, and passing the output to the following layer.

class Brain:
    def __init__(self, input_dim, output_dim, hidden_dim, network, learning_rate):
        self.learning_rate = learning_rate
        layer_mapping = {
            "Dense": Dense,
            "Tanh": Tanh,
            "Linear": Linear
        }
        self.network = []
        output = True
        for index, layer_type in enumerate(reversed(network)):
            layer_class = layer_mapping[layer_type]
            if layer_type == "Dense":
                if output:
                    self.network.insert(0, layer_class(hidden_dim, output_dim))
                    output = False
                elif index == len(network) -1:
                    self.network.insert(0, layer_class(input_dim, hidden_dim))
                else:
                    self.network.insert(0, layer_class(hidden_dim, hidden_dim))
            else:
                self.network.insert(0, layer_class())
           

    def forward(self, action):
        output = action
        for layer in self.network:
            output = layer.forward(output)

        return output
    
    def backward(self, error):
        grad = error
        for layer in reversed(self.network):
            grad = layer.backward(grad, self.learning_rate)
        return grad

# The ReplayBuffer class is just a manager for the agents' experience memory.

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    # This method stores the state, action, reward and next state in the replay buffer.
    def add_experience(self, state, action, reward, next_state):
        experience = (state, action, reward, next_state)
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            del self.buffer[0]

    def sample_batch(self, batch_size):
        # The choice method of numpy generates a list of indices.
        # The first argument is an array. If an int is given, the array is all the numbers from 0 to int_given
        # The second argument specifies how many items to pick from the given array (batch_size)
        # The replace argument determines if an item can be selected twice or not. 

        batch_indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]

        # The zip function groups the nth item of each batch. [[1,2], [a,b], [!, ?]] becomes [[1, a, !], [2, b, ?]]
        states, actions, rewards, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)