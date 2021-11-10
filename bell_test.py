import numpy as np 
from numpy import ndarray

class Bellman(object):

    def __init__(self, states: list, gamma: float =0.5):
        self.gamma = gamma 
        self.states = states

class Action(object):
    def __init__(self, probability, utility):
        self.probability: float = probability
        self.destination: int = utility

class State(object):

    def __init__(self,reward,utility,action_space):
        self.reward = reward
        self.utility = utility
        self.action_space = action_space
        self.next_state: State = self.set_consecutive()
    
    def set_consecutive(self):
        number_of_actions = len(self.action_space)
        action_space = []
        rewards = []
        consecutive_rewards = []
        consectutive_actions = []
        for i in range(number_of_actions):
            if i < round(number_of_actions/2):
                action_space.append(self.action_space[i])
                rewards.append(self.reward[i])
            else:
                consectutive_actions.append(self.action_space[i])
                consecutive_rewards.append(self.reward[i])

        self.action_space = action_space
        self.reward = rewards
        consecutive_state = []
        consecutive_state.append([consecutive_rewards,self.utility,consectutive_actions])
        return consecutive_state

# data splitting        
# 70 training, 15 test and 15 validation