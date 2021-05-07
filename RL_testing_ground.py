import numpy as np
import matplotlib.pyplot as plt

k = 100
iters = 1000
episodes = 50

class greedy_bandit:

    def __init__(self, arms, elipson, iterations, avg_arm_rewards):
        self.arms = arms
        self.elipson = elipson
        self.iterations = iterations
        self.steps = 0
        self.step_per_arm = np.zeros(arms)
        self.mean_reward = 0
        self.reward = np.zeros(iterations)
        self.mean_reward_per_arm = np.zeros(arms)  # predicted value, self.arm_rewards is true value

        if type(avg_arm_rewards) == list:
            self.arm_rewards = avg_arm_rewards
        elif avg_arm_rewards == "random":
            # ** samples taken from normal dist. ** np.random.normal (mean, standard dev., no. samples)
            self.arm_rewards = np.random.normal(0, 1, arms)
        elif avg_arm_rewards == "linspace":
            # **evenly spaced rewards, used for testing, higher arm no. higher reward **
            self.arm_rewards = np.linspace(0, arms - 1, arms)

    def pull(self):
        # make rand number to compare with elipson
        compare_probablity = np.random.rand()
        if self.elipson == 0 and self.steps == 0:  # random action to start off
            action = np.random.choice(self.arms)
        elif compare_probablity < self.elipson:  # explore
            action = np.random.choice(self.arms)
        else:  # exploit
            action = np.argmax(self.mean_reward_per_arm)

        reward = np.random.normal(self.arm_rewards[action], 1)

        self.steps += 1
        self.step_per_arm[action] += 1
        # based on analytical formula, saves indexing of all prev. values
        self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.steps
        # same thing below except updating the individual arm reward
        self.mean_reward_per_arm[action] += (reward - self.mean_reward_per_arm[action]) / self.step_per_arm[action]

    def iterate(self):
        for iteration in range(self.iterations):
            self.pull()
            self.reward[iteration] = self.mean_reward


class e_decay_bandit(greedy_bandit):
    def __init__(self, arms, decay_rate, elipson, iterations, avg_arm_rewards):
        greedy_bandit.__init__(self, arms, elipson, iterations, avg_arm_rewards)
        self.decay_rate = decay_rate

    def decay_pull(self):
        compare_probablity = np.random.rand()
        if self.elipson == 0 and self.steps == 0:  # random action to start off
            action = np.random.choice(self.arms)
        elif compare_probablity < self.elipson:  # explore
            action = np.random.choice(self.arms)
        else:  # exploit
            action = np.argmax(self.mean_reward_per_arm)

        reward = np.random.normal(self.arm_rewards[action], 1)

        self.steps += 1
        self.step_per_arm[action] += 1
        # based on analytical formula, saves indexing of all prev. values
        self.mean_reward += (reward - self.mean_reward) / self.steps
        # same thing below except updating the individual arm reward
        self.mean_reward_per_arm[action] += self.decay_rate * (reward - self.mean_reward_per_arm[action])

    def decay_iterate(self):
        for iteration in range(self.iterations):
            self.decay_pull()
            self.reward[iteration] = self.mean_reward


class ucb_bandit(greedy_bandit):
    def __init__(self, arms, c, iteras, avg_arm_rewards):
        greedy_bandit.__init__(self, arms, None, iteras, avg_arm_rewards)
        self.confidence_level = c
        self.t = np.zeros(arms)

    def ucb_pull(self):
        if self.steps == 0:  # random action to start off
            action = np.random.choice(self.arms)
        else:
            action = np.argmax(
                self.mean_reward_per_arm + (self.confidence_level * (np.sqrt(np.log(self.t) / self.step_per_arm))))

        reward = np.random.normal(self.arm_rewards[action], 1)

        self.steps += 1
        self.step_per_arm[action] += 1

        for index, arm in enumerate(self.t):
            if arm != action:
                self.t[index] += 1

        self.mean_reward += (reward - self.mean_reward) / self.steps

        self.mean_reward_per_arm[action] += (reward - self.mean_reward_per_arm[action]) / self.step_per_arm[action]

    def ucb_iterate(self):
        for iteration in range(self.iterations):
            self.ucb_pull()
            self.reward[iteration] = self.mean_reward

eps_01_rewards = np.zeros(iters)
eps_1_rewards = np.zeros(iters)
eps_decay_rewards = np.zeros(iters)
ucb_rewards = np.zeros(iters)

for i in range(episodes):
    random_test_list = [x for x in np.random.normal(0, 1, k)]

    eps_01 = greedy_bandit(k, 0.01, iters, random_test_list)
    eps_1 = greedy_bandit(k, 0.1, iters, random_test_list)
    eps_decay = e_decay_bandit(k, 0.3, 0.1, iters, random_test_list)
    ucb = ucb_bandit(k, 2, iters, random_test_list)

    eps_01.iterate()
    eps_1.iterate()
    eps_decay.decay_iterate()
    ucb.ucb_iterate()

    eps_01_rewards += (eps_01.reward - eps_01_rewards) / (i + 1)
    eps_1_rewards += (eps_1.reward - eps_1_rewards) / (i + 1)
    eps_decay_rewards += (eps_decay.reward - eps_decay_rewards) / (i + 1)
    ucb_rewards += (ucb.reward - ucb_rewards) / (i + 1)

plt.figure(figsize=(12, 8))
plt.plot(eps_01_rewards, label="epsilon=0.01")
plt.plot(eps_1_rewards, label="epsilon=0.1")
plt.plot(eps_decay_rewards, label="e_decay")
plt.plot(ucb_rewards, label="upper confidence bound")
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Avg Rewards after " + str(episodes) + " Episodes")
plt.show()
