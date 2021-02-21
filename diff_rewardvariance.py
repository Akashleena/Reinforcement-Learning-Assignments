import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

matplotlib.use('Agg')


class Bandit:
    # @k_arm: number of arms =20
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @Q0: denotes Q0 i.e initial estimation for each action we take :- Q0=0 and Q0=5
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    
    def __init__(self, k_arm=20, epsilon=0, Q0=0., sample_averages=False, UCB_param=None,
                 true_reward=0.):
        self.k = k_arm
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.Q0 = Q0

    def reset(self):
        # real reward for each action i.e true value of the action
        #q_true= q*(a) 
        self.q_true = np.random.randn(self.k) + self.true_reward 

        # estimation for each action
        #q_estimation=Qt(a)
        self.q_estimation = np.zeros(self.k) + self.Q0

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices) #generates a random sample from a given 1-D array

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(q*(At), 10) standard deviation 10
        reward = np.random.randn()*10 + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
    #######   Algorithm for finding the average reward for 1 bandit problem  #######
        self.average_reward += (reward - self.average_reward) / self.time 

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        
        return reward


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200, 20) + np.random.randn(20))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('/home/akashleena/Documents/20 arm bandit/Bandits_performance_plots/fig_Reward_distribution_rewardvar_10.png')
    plt.close()


def figure_2_2(runs=2000, time=1000):
    epsilons = [0, 0.05, 0.1, 0.5]
    bandits = [Bandit(epsilon=eps, sample_averages=True) for eps in epsilons] #sample_avg is set to true
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('/home/akashleena/Documents/20 arm bandit/Bandits_performance_plots/fig_egreedy_Avg_reward_optimal_action_rewardvar_10.png')
    plt.close()


if __name__ == '__main__':
    figure_2_1()
    figure_2_2()
