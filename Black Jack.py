import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict
import pygame

# Initialize game
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Black Jack")
font = pygame.font.Font(None, 36)

# Make enviroment
env = gym.make('Blackjack-v1', natural=False, sab=False)

# Learning Parameters
alpha = 0.2 # Learning Rate
gamma = 0.9 # Discount Factor
epsilon = 0.2 # Exploration probability
num_episodes = 100000

# Q table
default_q_value = 0.0
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Reinforcement Learning
results = {"Wins": 0, "Losses": 0, "Draws": 0}
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, done, _, _ = env.step(action)

        # Use of bellman equation
        best_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

        state = next_state

    # Results
    if reward == 1:
        results["Wins"] += 1
    elif reward == -1:
        results["Losses"] += 1
    else:
        results["Draws"] += 1

# Data Frame for results
df_results = pd.DataFrame([results])

# Watch the agent play
play_results = {"Wins": 0, "Losses": 0, "Draws": 0}
def play_bj(env, Q, episodes=100):
    running = True
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        print("New Game!")
        
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return  # Exit the function if the window is closed

            action = np.argmax(Q[state])
            next_state, reward, done, _, _ = env.step(action)
            
            # Display game state using pygame
            screen.fill((0, 128, 0))
            state_text = font.render(f"Player Sum: {state[0]}, Dealer Card: {state[1]}", True, (255, 255, 255))
            action_text = font.render(f"Action: {'Hit' if action == 1 else 'Stay'}", True, (255, 255, 0))
            screen.blit(state_text, (50,100))
            screen.blit(action_text, (50,150))
            pygame.display.update()

            pygame.time.delay(5)

            state = next_state

        if reward == 1:
            play_results["Wins"] += 1
        elif reward == -1:
            play_results["Losses"] += 1
        else:
            play_results["Draws"] += 1

        print("Game Over!\n")

play_bj(env, Q)

# PLOT
plt.figure(figsize=(8, 5))
plt.bar(["Wins", "Losses", "Draws"], [play_results["Wins"], play_results["Losses"], play_results["Draws"]], color=["green", "red", "blue"])
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.title("Blackjack Results after 100 Games")
plt.show()

#quit
pygame.quit()