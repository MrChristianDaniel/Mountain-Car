# Q Learning
# Continuous, Finite State Space
# Discrete, Finite Action Space

import gym
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def policy(state, e_rate):
    if random.random() > e_rate:
        return np.argmax(Q[state[0], state[1]])
    else:
        return env.action_space.sample()

env = gym.make('MountainCar-v0', render_mode = 'rgb_array')

episode_max = 100
episode_min = 0
learning_rate = 0.2
discount = 0.9
exploration_rate = 0.8
exploration_decay = (exploration_rate - episode_min)/episode_max
frames_beg = []
frames_end = []
reward_array = []

state_space = [50, 50]
action_space = 3

bins = [np.linspace(-1.2, 0.6, state_space[0]),
        np.linspace(-0.07, 0.07, state_space[1])]

Q = np.zeros((state_space[0], state_space[1], action_space))

for i in range(episode_max):
    done = False
    total_reward = 0

    state = env.reset()
    state_discrete = np.zeros(len(state[0]))

    for j in range(len(state[0])):
        state_discrete[j] = np.digitize(state[0][j], bins[j])

    state_discrete = np.round(state_discrete, 0).astype(int)

    if i == 1:
        frame = env.render()
        frames_beg.append(Image.fromarray(frame))
    elif i == episode_max - 1:
        frame = env.render()
        frames_end.append(Image.fromarray(frame))

    while not done:
        action = policy(state_discrete, exploration_rate)

        state2, reward, done, trunc, info = env.step(action)
        state2_discrete = np.zeros(len(state2))

        for j in range(len(state2)):
            state2_discrete[j] = np.digitize(state2[j], bins[j])

        state2_discrete = np.round(state2_discrete, 0).astype(int)

        frame = env.render()
        if i == 1:
            frames_beg.append(Image.fromarray(frame))
        elif i == episode_max - 1:
            frames_end.append(Image.fromarray(frame))

        if done and state2[0] >= 0.5:
            Q[state_discrete[0], state_discrete[1], action] = reward
        else:
            delta = learning_rate * (reward + discount * np.max(Q[state2_discrete[0], state2_discrete[1]]) - Q[state_discrete[0], state_discrete[1], action])
            Q[state_discrete[0], state_discrete[1], action] += delta

        state_discrete = state2_discrete
        total_reward += reward

    print('Episode:', i+1, 'Reward:', total_reward)

    reward_array.append(total_reward)

    if i > episode_min:
        exploration_rate -= exploration_decay

plt.plot(reward_array)
plt.title('MountainCar Agent')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.savefig('MountainCar_Agent_Graph.png')

frames_beg[0].save('MountainCar_Agent_Beginning.gif', save_all = True, append_images=frames_beg[1:], optimize = False, duration = 40, loop = 0)
frames_end[0].save('MountainCar_Agent_Ending.gif', save_all = True, append_images=frames_end[1:], optimize = False, duration = 40, loop = 0)

env.close()