import gym
import time
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


env = gym.make('MsPacman-v0')
state = env.reset()

epoch = 40

rewards = []

accreward = 0
index = 0

while(index < epoch):
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    if reward > 0:
        accreward += reward
    if done:
        env.reset()
        rewards.append(accreward)
        print(accreward)
        accreward = 0
        index += 1
        
env.close()


plt.figure(figsize=(12, 8))

plt.subplot(221)

r = np.array(rewards)
mean = np.mean(r)
m = np.max(r)
plt.title("Accumulated Rewards; Mean: %.2f; Max: %.2f"%(mean, m))
plt.plot(rewards)

plt.subplot(223)

ecdf = sm.distributions.ECDF(r)

x = np.linspace(np.min(r), np.max(r), num=100)
y = ecdf(x)
plt.title("rewards ECDF")
plt.plot(x, y)
plt.plot([np.mean(r)], [ecdf(np.mean(r))], 'o')

plt.text(np.mean(r), ecdf(np.mean(r)), 'P(reward > %.2f) = %.3f'%(np.mean(r), 1-ecdf(np.mean(r))))

plt.savefig('1.jpg')