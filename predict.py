import os
import time
import numpy as np

from memory import Memory
from net import Net
from env import Env
from train import Train

import statsmodels.api as sm
import matplotlib.pyplot as plt

#修改进程工作目录
PROCESS_PATH = '/Users/newwenshun/Desktop/数据工程实践/project/'
os.chdir(PROCESS_PATH)


env = Env('MsPacman-v0')
env.reset()

memory = Memory(80)

path = 'results/moduleV3.pkl'
train = Train(env, memory, path)


epoch = 40

rewards = []
accreward = 0.0

index = 0

while(index < epoch):
    reward, done = train.predict()
    accreward += reward
    if done:
        rewards.append(accreward)
        print(accreward)
        accreward = 0
        index += 1
        train.reset()

plt.figure(figsize=(12, 4))

plt.subplot(121)

r = np.array(rewards)
mean = np.mean(r)
m = np.max(r)
plt.title("Accumulated Rewards; Mean: %.2f; Max: %.2f"%(mean, m))
plt.plot(rewards)

plt.subplot(122)

ecdf = sm.distributions.ECDF(r)

x = np.linspace(np.min(r), np.max(r), num=100)
y = ecdf(x)
plt.title("rewards ECDF")
plt.plot(x, y)
plt.plot([np.mean(r)], [ecdf(np.mean(r))], 'o')

plt.text(np.mean(r), ecdf(np.mean(r)), 'P(reward > %.2f) = %.3f'%(np.mean(r), 1-ecdf(np.mean(r))))

plt.savefig('predict.jpg')