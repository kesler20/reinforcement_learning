
from bell_test import *
import numpy as np
from kernels import Kernel 
from test import data_frame, observed, predicted

#reward = Kernel(observed, predicted)
#reward = reward.gaussian_kernel

action_space = [data_frame[0].fillna(0)]
print(action_space)
state = State(np.array([i for i in range(len(action_space))]),2,action_space)
consecutive_state = state.set_consecutive()

states = [state, consecutive_state]
bell = Bellman(states)

previous_utility = -10000
diff = 1000000
error = 0.000002
while diff > error:
    actions = [[i*states[0].utility for i in states[0].action_space], [i*states[0].utility for i in states[1].action_space]]
    consecutive_utility = states[0].reward + 0.5*np.argmin([sum(actions[0],sum(actions[1]))])
    diff = np.abs(consecutive_utility - previous_utility)
    print(diff)
    previous_utility = consecutive_utility
print(previous_utility)
