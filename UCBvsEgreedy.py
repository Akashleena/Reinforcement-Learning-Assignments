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
    
    def __init__(self, k_arm=20, epsilon=0, Q0=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
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
        # generate the reward under N(q*(At), 1) standard deviation 1
        reward = np.random.randn()*1 + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
    #######   Algorithm for finding the average reward for 1 bandit problem  #######
        self.average_reward += (reward - self.average_reward) / self.time 

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
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
    plt.savefig('/home/akashleena/Documents/20 arm bandit/Bandits_performance_plots/fig_Reward_distribution_UCBvsepsilongreedy.png')
    plt.close()


def figure_2_2(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(epsilon=0, UCB_param=1, sample_averages=True))
    bandits.append(Bandit(epsilon=0, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0, UCB_param=5, sample_averages=True))
  
    bandits.append(Bandit(epsilon=0.0, sample_averages=True))
    bandits.append(Bandit(epsilon=0.05, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, sample_averages=True))
    bandits.append(Bandit(epsilon=0.5, sample_averages=True))
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label='UCB c = 1')
    plt.plot(average_rewards[1], label='UCB c = 2')
    plt.plot(average_rewards[2], label='UCB c = 5')
    plt.plot(average_rewards[3], label='greedy epsilon= 0.0')
    plt.plot(average_rewards[4], label='epsilon greedy = 0.05')
    plt.plot(average_rewards[5], label='epsilon greedy = 0.1')
    plt.plot(average_rewards[6], label='epsilon greedy= 0.5')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('/home/akashleena/Documents/20 arm bandit/Bandits_performance_plots/Avg reward_UCB_vs_epsilon greedy.png')
    plt.close()

def figure_2_3(runs=2000, time=1000):
    labels = ['epsilon-greedy',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sample_averages=True),
                  lambda coef: Bandit(epsilon=0, UCB_param=coeff, sample_averages=True),
                  lambda Q0: Bandit(epsilon=0, Q0=Q0, step_size=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = []
    for generator, parameter in zip(generators, parameters):
        for param in parameter:
            bandits.append(generator(pow(2, param)))

    _, average_rewards = simulate(runs, time, bandits)
    rewards = np.mean(average_rewards, axis=1)

    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i+l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('/home/akashleena/Documents/20 arm bandit/Bandits_performance_plots/UCB_vs_Egreedy_vs_optimisticinitialization.png')
    plt.close()

if __name__ == '__main__':
    figure_2_1()
    figure_2_2()
    figure_2_3()

