# Reinforcement Learning: Blackjack Q-Learning Agent

## What is Reinforcement Learning (RL)?
Reinforcement Learning (RL) is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The process follows these steps:

1. **Observation**: The agent perceives the current state of the environment.
2. **Action**: It chooses an action based on its strategy.
3. **Reward**: The agent receives a reward or penalty based on the chosen action.
4. **Update**: It adjusts its strategy to improve future decisions.

This cycle repeats until the agent learns an **optimal strategy** that maximizes total long-term rewards.

---

## Blackjack Q-Learning Agent

### Goal
The goal of this project is to train a **Reinforcement Learning agent** to play Blackjack using the **Q-learning** method. We will implement an **epsilon-greedy** strategy for action selection and evaluate the model's performance after training.

---

## The Bellman Equation for Q-Value Updates

Q-learning uses the **Bellman equation** to update $Q(s, a)$  values, which represent the expected utility of taking action $a$ in state $s$:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

where:
- $Q(s, a)$ represents the value of taking action $a$ in state $s$.
- $\alpha$ (learning rate) determines how much new experiences influence updates.
- $r$ is the reward obtained after taking action $a$.
- $\gamma$ (discount factor) controls the importance of future rewards.
- $\max_{a'} Q(s', a')$ is the estimated value of the best action in the next state $s'$.

---

## Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import defaultdict
```
---

### Configuration

The script uses the following hyperparameters:

- **alpha (0.4)**: Learning rate.
- **gamma (0.9)**: Discount factor for future rewards.
- **epsilon (0.4)**: Exploration probability in the epsilon-greedy policy.
- **train_episodes (250000)**: Number of training episodes.
- **test_episodes (50)**: Number of test episodes after training.

---

## How the Code Works

### Initialization

- The Blackjack environment is set up using `gymnasium`.
- A **Q-table** is created to store state-action values.

### Epsilon-Greedy Policy

- The agent chooses a **random action** with probability $\epsilon$.
- Otherwise, it picks the **best-known action** based on the Q-table.

### Training

- The agent plays **250,000 Blackjack games** to learn optimal strategies.
- The **Q-table** is updated using the **Bellman equation** during each game.

### Testing

- After training, the agent plays **50 test games** without exploration, always selecting the best action (exploitation).

### Visualization

- The agent's performance is visualized by plotting the number of **wins, losses, and draws** across the test games.

### Closing the Environment

- The **gymnasium environment** is properly closed after the testing phase to release resources.

---

### Additional Considerations

In real life, casinos use multiple decks in Blackjack to make card counting more difficult. It is recommended to modify the simulation to include different numbers of decks and analyze how it affects the agent's strategy:

* **4-Deck Stack**:  
  The agent plays with a 4-deck stack, which increases the complexity of the environment and allows the agent to learn more advanced strategies.
  
* **Card Counting**:  
  The agent implements a basic card counting technique (Hi-Lo) to determine whether it's favorable to draw an additional card. A counter is used, which increases with low cards (2-6) and decreases with high cards (10, J, Q, K, A), helping the agent determine if the odds are in its favor to draw another card.

---
#### Authors:
- Rania Aguirre
- Manuel Reyna
