import gymnasium as gym
import numpy as np
import random
from gymnasium.wrappers import TransformReward
import matplotlib.pyplot as plt

def rewardChange(r,a,b,c):
    if (r==-1):
        return a
    elif r==20:
        return b
    else :
        return c

ar=-1
br=20
cr=-10



plt.figure(figsize=(10,6))
plt.xlabel("Episodes")
plt.ylabel("Rewards Per 100 episodes")

"""Setup"""

def run(a,b,c,cnt):

    env = gym.make("Taxi-v3")
    env = TransformReward(env, lambda r:rewardChange(r,a,b,c))

    # Setup the Gym Environment


    # Make a new matrix filled with zeros.
    # The matrix will be 500x6 as there are 500 states and 6 actions.
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    training_episodes = 20000 # Amount of times to run environment while training.

    # Hyperparameters
    alpha = 0.1 # Learning Rate
    gamma = 0.6 # Discount Rate
    epsilon = 0.1 # Chance of selecting a random action instead of maximising reward.

   

    """Training the Agent"""

    penalty_episodic=np.zeros(training_episodes)

    for i in range(training_episodes):
        state = env.reset()[0] # Reset returns observation state and other info. We only need the state.
        done = False
        penalties, reward, = 0, 0
        
        while not done:
            if random.uniform(0, 1) < epsilon :
                action = env.action_space.sample() # Pick a new action for this state.
            else:
                action = np.argmax(q_table[state]) # Pick the action which has previously given the highest reward.

            next_state, reward, done,_, info = env.step(action)
            # print(env.step(action))
            
            old_value = q_table[state, action] # Retrieve old value from the q-table.
            next_max = np.max(q_table[next_state])

            # Update q-value for current state.
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == c: # Checks if agent attempted to do an illegal action.
                penalties += 1

            state = next_state

        penalty_episodic[i]=penalties
        if i % 5000 == 0: # Output number of completed episodes every 100 episodes.
            print(f"Episode: {i}")

    print(str(cnt)+" done!!!")

    sum_rewards = np.zeros(training_episodes)
    for t in range(training_episodes):
        sum_rewards[t] = np.sum(penalty_episodic[0:(t+1)])
    # plt.clf()

    
    plt.plot(sum_rewards,label="("+str(a)+","+str(b)+","+str(c)+")")

    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.tight_layout()
    plt.savefig("taxicomp.png")

    # plt.savefig("taxi4p"+str(cnt)+".png")


"""Display and evaluate agent's performance after Q-learning."""

total_epochs, total_penalties = 0, 0


for i in range(2,8):
    run(-1,50,-1*(i),i)


for i in range(10,51,5):
    run(-1,50,-1*(i),i)



