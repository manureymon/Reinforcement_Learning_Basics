import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict

# Configuration
alpha = 0.4  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.4  # Exploration probability
train_episodes = 250000  
test_episodes = 50 

# ENVIROMENT FOR TRAIN
env = gym.make('Blackjack-v1')

# Initialize Q-table
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Function to choose action using epsilon-greedy policy
def select_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        return np.argmax(Q[state])  # Exploitation

# TRAINING
for episode in range(train_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)

        # Apply Bellman equation to update Q-values
        best_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

        state = next_state

# CREATE ENVIRONMENT WITH RENDERING FOR TESTING
env = gym.make('Blackjack-v1', render_mode="human")

# Play
test_results = {"Wins": 0, "Losses": 0, "Draws": 0}

for _ in range(test_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(Q[state])  # Best action
        state, reward, done, _, _ = env.step(action)

    # Record results
    if reward == 1:
        test_results["Wins"] += 1
    elif reward == -1:
        test_results["Losses"] += 1
    else:
        test_results["Draws"] += 1

# DISPLAY RESULTS
plt.figure(figsize=(8, 5))
plt.bar(test_results.keys(), test_results.values(), color=["green", "red", "blue"])
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.title(f"Blackjack Results after {test_episodes} Games")
plt.show()

# CLOSE THE ENVIRONMENT
env.close()