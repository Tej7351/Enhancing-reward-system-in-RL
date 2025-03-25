import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TransformReward

num=0


def rewardChange(r,a,b):
  if (r==0):
    return a
  else:
    return b
  
plt.figure(figsize=(10,6))
plt.xlabel("Episodes")
plt.ylabel("Cumulative Count of Achieving Goal")

def run(env,color,label,episodes,a,b,rew=1, is_training=True, render=False):
    
    global num
    
    q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
        
    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 0.2         # 1 = 100% random actions
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)


    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200
        cnt=0

        while(not terminated and not truncated):
            if is_training and ( rng.random() < epsilon ):
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if  terminated and reward!=b:
                reward=a*10

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state
            cnt+=1
            
        if i%10000==0:
            print(str(i)+" "+str(a)+" "+str(b))
           
            

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == b:
            rewards_per_episode[i] = 1
        
    
    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[0:(t+1)])
    # plt.clf()
    plt.plot(sum_rewards,color=color,label=label)
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.tight_layout()
    # num+=1
    # plt.savefig('f_l4x41'+str(num)+'.png')
    plt.savefig('Total_Success.png')



   





def run_env(a,b,color):
    # num=0
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
    env = TransformReward(env, lambda r:rewardChange(r,a,b))
    run(env=env,color=color,label="("+str(a)+","+str(b)+")",a=a,b=b,episodes=60000)




if __name__ == '__main__':
    
    run_env(0, 1, "darkred")
    run_env(0, 5, "lightcoral")
    run_env(0, 10, "orange")
    run_env(0, 50, "mediumvioletred")
    run_env(0, 100, "palevioletred")
    run_env(-1, 0, "yellow")
    run_env(-1, 1, "lime")
    run_env(-1, 20, "forestgreen")
    run_env(-10, 0, "skyblue")
    run_env(-10, 1, "navy")
    run_env(-50, 0, "deepskyblue")
    run_env(-50, 1, "indigo")
    run_env(-200, 1, "purple")




    





  