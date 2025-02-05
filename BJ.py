import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
import gymnasium as gym
import pygame

# Make enviroment
env = gym.make('Blackjack-v1', render_mode="rgb_array")

# Reinforcement Learning
for i in range(100):
    state = env.reset()
    while True:
        print(state)
        action = env.action_space.sample()
        state, reward, done, _, _ = env.step(action)
        if done:
            print('End game! Reward:', reward)
            print('You won!') if reward > 0 else print('You lost!')
            break

# Reinforcement Learning
for i in range(100):
    state = env.reset()
    episode = []
    while True:
        action = 0 if  20 > 18 else 1
        next_state, reward, done, _, _  = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            print('End game! Reward:', reward)
            print('You won!') if reward > 0 else print('You lost!')
            break

# Game
env.reset()
pygame.init()
prev_screen = env.render()
plt.imshow(prev_screen)
num_episodes = 1000
dis = 0.99
rList = []

for i in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

for i in range(100):
    action = 0 if 20 > 18 else 1
    next_state, reward, done, _, _ = env.step(action)
    episode.append((state, action, reward))
    state = next_state

    screen = env.render()

    plt.imshow(screen)
    ipythondisplay.clear_output(wait=True)
    ipythondisplay.display(plt.gcf())

    if done:
        break

    ipythondisplay.clear_output(wait=True)
    env.close()


print("Average reward over {} episodes: {}".format(num_episodes, np.mean(rList)))