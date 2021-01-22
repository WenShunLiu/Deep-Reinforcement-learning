import os
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

print('Projet -- net default')

epoch = 40

memory = Memory(80)
env = Env('MsPacman-v0')
env.reset()

path = 'results/moduleV3.pkl'
train = Train(env, memory, path)

rewards = []

accreward = 0
index = 0
while(index < epoch):
    action, reward, done = train.train()
    if reward > 0:
        accreward += reward
    if done:
        train.reset()
        rewards.append(accreward)
        accreward = 0
        index += 1
train.saveModule()



# 保存奖励图与loss图
PATH = 'rewards/'
MAX_REWARD = 30

if not os.path.isdir(PATH):
    os.mkdir(PATH)

files = os.listdir(PATH)

files.sort(key = lambda x: int(x[0:-4]))

if len(files) >= MAX_REWARD:
    os.remove(PATH + files[0])


filename = 1
if len(files) != 0:
    lastname = int(files[-1][0:-4])
    filename = lastname + filename

filename = str(filename) + '.jpg'


plt.figure(figsize=(12, 8))

plt.subplot(221)

r = np.array(rewards)
mean = np.mean(r)
m = np.max(r)
plt.title("Accumulated Rewards; Mean: %.2f; Max: %.2f"%(mean, m))
plt.plot(rewards)

plt.subplot(222)

l = np.array(train.loss_array)
mean = np.mean(l)
plt.title("LOSS; Mean: %.10f"%(mean))
plt.plot(train.loss_array)

plt.subplot(223)

ecdf = sm.distributions.ECDF(r)

x = np.linspace(np.min(r), np.max(r), num=100)
y = ecdf(x)
plt.title("rewards ECDF")
plt.plot(x, y)
plt.plot([np.mean(r)], [ecdf(np.mean(r))], 'o')

plt.text(np.mean(r), ecdf(np.mean(r)), 'P(reward > %.2f) = %.3f'%(np.mean(r), 1-ecdf(np.mean(r))))

plt.savefig(PATH + filename)

#plt.show()
