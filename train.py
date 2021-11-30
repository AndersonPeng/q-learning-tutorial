import gym
import numpy as np
from matplotlib import pyplot as plt
from collections import deque

#Get discrete state: return the corresponding index in q_table
def get_discrete_state(state, bins, s_dim):
    state_idx = [0] * s_dim

    for i in range(s_dim):
        state_idx[i] = np.digitize(state[i], bins[i]) - 1

    return tuple(state_idx)

#Main function
def main():
    #Parameters
    n_episode   = 20000   #Number of episodes
    n_disp      = 1000    #Number of steps to display
    n_bins      = 16      #Number of bins
    lr          = 0.1     #Learning rate
    gamma       = 0.95    #Discount factor (gamma)
    eps         = 1.0     #Epsilon
    buffer_size = 512     #Replay byffer size
    mb_size     = 4       #Mini-batch size

    #Create environment
    env = gym.make('CartPole-v0')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    print(s_dim)
    print(a_dim)

    #Create Q-table: (n_bins, n_bins, n_bins, n_bins, a_dim)
    bins = [
        np.linspace(-2.4, 2.4, n_bins),     #Cart position
        np.linspace(-4.0, 4.0, n_bins),     #Cart velocity
        np.linspace(-0.418, 0.418, n_bins), #Pole angle
        np.linspace(-4.0, 4.0, n_bins)      #Pole velocity at tip
    ]
    q_table = np.zeros(([n_bins] * s_dim + [a_dim]), dtype=np.float32)

    #Start training
    total_rewards = np.zeros((n_episode+1), dtype=np.float32)
    replay_buffer = deque(maxlen=buffer_size)

    #For each episode
    for i_episode in range(n_episode+1):
        state = env.reset()
        discrete_state = get_discrete_state(state, bins, s_dim)
        total_reward = 0

        #For each time step
        while True:
            #Choose action (epsilon greedy): a = argmax_a' Q(s, a')
            if np.random.random() > eps:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, a_dim)
            
            #Interact with the environment: put (s, a, r, s') to the replay buffer
            next_state, reward, done, _ = env.step(action)
            next_discrete_state = get_discrete_state(next_state, bins, s_dim)
            replay_buffer.append((discrete_state, action, reward, next_discrete_state))
            discrete_state = next_discrete_state
            total_reward += reward

            #Update Q-table: Q(s, a) <- Q(s, a) + lr * (r(s, a) + gamma * max_a' Q(s', a') - Q(s, a))
            if len(replay_buffer) >= mb_size:
                for i in np.random.choice(len(replay_buffer), mb_size, replace=False):
                    s, a, r, next_s = replay_buffer[i]
                    max_next_q = np.max(q_table[next_s])
                    current_q = q_table[s + (a,)]
                    q_table[s + (a,)] = current_q + lr * (reward + gamma*max_next_q - current_q)

            #Record total reward
            if done:
                total_rewards[i_episode] = total_reward
                break
        
        #Decay epsilon exponentially
        eps = np.exp(-0.2 - 4.0*i_episode / n_episode)

        #Print the current result
        if i_episode % n_disp == 0:
            print("[{:5d}/{:5d}] total_reward = {:.3f}, avg_total_reward = {:.3f}, epsilon = {:.3f}".format(
                i_episode, n_episode, total_reward, total_rewards[:i_episode+1].mean(), eps
            ))

    #Save Q-table
    with open("qtable.npy", "wb") as f:
        print("Saving the Q-table ... ", end="")
        np.save(f, q_table)
        print("Done.")

    env.close()

    #Total reward plot
    window_size = 50
    episodes = np.arange(0, n_episode)
    mean_total_rewards = np.zeros((n_episode))
    std_total_rewards = np.zeros((n_episode))

    for i in range(n_episode):
        left = max(0, i - window_size)
        right = min(len(total_rewards), i + window_size)
        mean_total_rewards[i] = total_rewards[left:right].mean()
        std_total_rewards[i] = total_rewards[left:right].std()

    plt.xlim(0, n_episode)
    plt.ylim(0, 200)
    plt.grid()
    plt.plot(episodes, mean_total_rewards, color='red')
    plt.fill_between(
        episodes, 
        mean_total_rewards+std_total_rewards, 
        mean_total_rewards-std_total_rewards, 
        color='red', 
        alpha=0.4
    )
    plt.show()

if __name__ == '__main__':
    main()
