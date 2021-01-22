import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
from collections import namedtuple

class Memory(object):
    def __init__(self, maxSize):
        super(Memory, self).__init__()
        self.maxSize = maxSize
        self.data = []
        self.Item = namedtuple("Item", ["state", "action", "reward", "next_state", "done", "info"])
    def add(self, state, action, reward, next_state, done, info):
        if(self.getLen() >= self.maxSize):
            self.data.pop(0)
        item = self.Item(state, action, reward, next_state, done, info)
        self.data.append(item)


    def getSample(self, batch):
        if self.getLen() < batch:
            print("the amount of data is not enough")
            return
        index = random.sample(range(self.getLen()), batch)
        state = []
        action = []
        reward = []
        next_state = []
        done = []
        info = []
        for i in index:
            state.append(self.data[i].state)
            action.append([self.data[i].action])
            reward.append([self.data[i].reward])
            next_state.append(self.data[i].next_state)
            done.append([self.data[i].done])
            info.append([self.data[i].info])

        self.data = []
        return np.array(state), np.array(action, dtype=np.int8), np.array(reward, dtype=np.float16), np.array(next_state), np.array(done, dtype=np.int8), np.array(info, dtype=object)

    def getLen(self):
        return len(self.data)




'''
state: 当前状态
action: 执行动作
reward: 奖励
next_state: 下一个状态
done: 是否结束
info: 状态信息
'''