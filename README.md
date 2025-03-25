# Exploring Safe and Effective Reward Patterns in Reinforcement Learning

## Overview
This repository contains the implementation and analysis of reward systems in Reinforcement Learning (RL), specifically focusing on the **Frozen Lake Problem** and the **Taxi Problem** using the **Gymnasium (or Gym) library**. The primary objective is to evaluate different reward structures to determine their impact on agent performance and safety in decision-making.

## Repository Structure
The repository is organized as follows:

### **Frozen Lake Problem**
This directory includes Python scripts implementing **Q-learning** on the Frozen Lake environment, comparing different reward structures, including action-penalty and goal-based rewards, using reward wrappers in Gymnasium.

- **`Frozen_lake_steps.py`**: Implements Q-learning on the Frozen Lake environment and plots the number of steps taken under different reward structures against the number of episodes.
- **`Frozen_lake_success.py`**: Implements Q-learning on the Frozen Lake environment and plots cumulative success rates for different reward structures over episodes.

### **Taxi Problem**
This directory contains Python scripts analyzing safe reward structures in the **Taxi Problem**, an environment where illegal moves can occur.

- **`taxi_penalty_count.py`**: Implements Q-learning on the Taxi environment and evaluates the impact of penalties for illegal moves by plotting the number of illegal actions taken across episodes.

## Results and Insights
### **Frozen Lake Problem**
- The analysis demonstrates that an **action-penalty reward system** improves learning efficiency and stability compared to a purely goal-based reward system in the Frozen Lake environment.

### **Taxi Problem**
- The findings indicate that a **penalty for illegal moves must be sufficiently higher than the reward for on-route actions** to ensure safe and efficient learning in environments with potential violations.

## Usage Instructions
To execute the scripts, ensure you have Python and the Gymnasium library installed. Run the respective Python scripts in each directory to generate and visualize results.

### **Prerequisites**
- Python (>=3.7 recommended)
- Gymnasium (or Gym) library

Run the scripts using:
```bash
python Frozen_lake_Problem/Frozen_lake_steps.py
python Frozen_lake_Problem/Frozen_lake_success.py
python Taxi_Problem/taxi_penalty_count.py
```
This will generate corresponding plots and insights on the impact of different reward structures on learning performance.

---
This project aims to contribute to the understanding of **reward design in reinforcement learning** and its implications for agent behavior in dynamic environments.
