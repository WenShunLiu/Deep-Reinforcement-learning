import gym
import time

env = gym.make('MsPacman-v0')
env2 = gym.make('MsPacman-ram-v0')
ob = env.reset()
ob2 = env2.reset()
print(env.observation_space)
print(env.action_space.n)
#print(ob)
count = 0

# 原地
for i in range(85):
    #env.render()
    env2.render()
    ob, reward, done, info = env.step(0)
    ob2, reward2, done2, info2 = env2.step(0)
    print(reward)
    if(reward == 10):
        count += 1
# 向左
for i in range(30):
    #env.render()
    env2.render()
    ob, reward, done, info = env.step(3)
    ob2, reward2, done2, info2 = env2.step(0)
    print(reward)
    if(reward == 10):
        count += 1
    time.sleep(0.1)

# 向下
for i in range(5):
    #env.render()
    env2.render()
    ob, reward, done, info = env.step(4)
    ob2, reward2, done2, info2 = env2.step(0)
    print(reward)
    print(info)
    if(reward == 10):
        count += 1
    time.sleep(0.1)

# 向左
for i in range(17):
    #env.render()
    env2.render()
    ob, reward, done, info = env.step(3)
    ob2, reward2, done2, info2 = env2.step(0)
    print(reward)
    print(ob)
    if(reward == 10):
        count += 1
    time.sleep(0.1)

# 向下
for i in range(10):
    #env.render()
    env2.render()
    ob, reward, done, info = env.step(4)
    ob2, reward2, done2, info2 = env2.step(0)
    print(reward)
    #print(ob)
    if(reward == 10):
        count += 1
    time.sleep(0.1)
print(count)
for i in range(2000):
    #env.render()
    env2.render()
env.close()
