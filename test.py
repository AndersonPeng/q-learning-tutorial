import gym
import numpy as np

#Get discrete state: return the corresponding index in q_table
def get_discrete_state(state, bins, s_dim):
    state_idx = [0] * s_dim

    for i in range(s_dim):
        state_idx[i] = np.digitize(state[i], bins[i]) - 1

    return tuple(state_idx)

#Main function
def main():
    #Parameters
    n_episode = 2
    n_bins = 16

    #Create environment
    env = gym.make('CartPole-v0')
    s_dim = len(env.observation_space.high)
    a_dim = env.action_space.n

    #Create Q-table: (n_bins, n_bins, n_bins, n_bins, a_dim)
    bins = [
        np.linspace(-2.4, 2.4, n_bins),     #Cart position
        np.linspace(-4.0, 4.0, n_bins),     #Cart velocity
        np.linspace(-0.418, 0.418, n_bins), #Pole angle
        np.linspace(-4.0, 4.0, n_bins)      #Pole velocity at tip
    ]
    q_table = np.random.uniform(low=-2, high=0, size=([n_bins] * s_dim + [a_dim]))

    #Load Q-table
    with open("qtable.npy", "rb") as f:
        q_table = np.load(f)

    #Start testing
    #For each episode
    for i_episode in range(n_episode):
        state = env.reset()
        discrete_state = get_discrete_state(state, bins, s_dim)
        total_reward = 0
        episode_len = 0

        #For each time step
        while True:
            env.render()

            #1. Choose action: a = argmax_a' Q(s, a')
            action = np.argmax(q_table[discrete_state])
            
            #2. Interact with the environment: (s, a, r, s')
            next_state, reward, done, _ = env.step(action)
            next_discrete_state = get_discrete_state(next_state, bins, s_dim)

            #3. Update state
            discrete_state = next_discrete_state
            total_reward += reward
            episode_len += 1
            
            if done:
                print("[{:2d}/{:2d}] length = {:d}, total_reward = {:f}".format(
                    i_episode+1, n_episode, episode_len, total_reward
                ))
                break

    env.close()

if __name__ == '__main__':
    main()
