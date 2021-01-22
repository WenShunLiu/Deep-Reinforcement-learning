import gym
import numpy as np

class Env(object):
    def __init__(self, ID):
        self.env = gym.make(ID)
        self.actionSize = self.env.action_space.n
        self.actionSet = self.env.getActionSet()

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def step(self, action):
        return self.env.step(action)
        
    def getState(self):
        return self.env.get_obs()

    def getActionSample(self):
        return self.env.action_space.sample()

    def close(self):
        return self.env.close()

    def getReward(self):
        return {
            default: 0.0,
            smallPac: 10.0,
            bigPac: 50.0,
            death: 0.0
        }

    def getActionSize(self):
        return self.actionSize

    def getActionSet(self):
        return self.actionSet






'''
reward：
    普通状态：0
    吃到小豆子：10
    吃到大豆子：50
    死掉：0

info： 还剩ale.lives条命

_action_set = [0 2 3 4 5 6 7 8 9]

ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
'''
