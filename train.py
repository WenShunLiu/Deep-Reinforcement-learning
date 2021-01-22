import random
import math
import torch
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F

from torch import optim
from memory import Memory
from net import Net
from env import Env

class Train(object):

    epsilon = 1.0
    gamma = 0.99
    target = None
    net = None
    lossFn = None
    lr = 0.000001
    optimizer = None

    preLives = 3

    loss_array = []

    step = 1

    step_loss = []

    def __init__(self, env, memory, path):
        self.env = env
        self.memory = memory
        self.path = path
        self._buildNet()

    def _buildNet(self):
        actionSize = self.env.getActionSize()
        self.target = Net(actionSize)
        self.net = Net(actionSize)
        
        if os.path.isfile(self.path):
            print('module exit! Train beginning')
            self.epsilon = 0.55
            self.net.load_state_dict(torch.load(self.path))
        else:
            print('module do not exit! Train beginning from 0')

        self.targetReplace()
        self.lossFn = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.net.parameters(), self.lr)

    def reset(self):
        self.preLives = 3
        self.env.reset()


    def _chooseAction(self, state):
        state = np.array([state])
        if random.random() < self.epsilon:
            action = self.env.getActionSample()
            return action
        else:
            with torch.no_grad():
                q_next = self.net(state)
            _, index = torch.max(q_next, 1)
            action = index.squeeze(0).detach().item()
            return action

    def train(self):
        #self.env.render()
        state = self.env.getState()
        action = self._chooseAction(state)
        next_state, reward, done, info = self.env.step(action)

        reward1 = -1.0
        life = list(info.values())[0]
        if done:
            reward1 = -5.0
        elif life < self.preLives:
            reward1 = -2.5
        elif reward == 10.0:
            reward1 = 0.5
        elif reward > 10.0:
            reward1 = math.sqrt(reward) / 10

        self.preLives = life

        self.memory.add(state, action, reward1, next_state, done, info)

        self._learn()
        self.epsilon = max(0.1, self.epsilon * 0.85)

        return action, reward, done

    def _learn(self, batch = 32):
        if self.memory.getLen() < batch:
            return
        state, action, reward, next_state, done, info  = self.memory.getSample(batch)

        action = torch.from_numpy(action).long()
        reward = torch.from_numpy(reward).float()
        done = torch.from_numpy(done)

        Q_target_next = self.target(next_state).detach().max(1)[0].unsqueeze(1)

        Q_target = reward + self.gamma * Q_target_next * (torch.ones_like(done) - done)

        Q_expect = self.net(state).gather(dim=1, index=action)

        self.optimizer.zero_grad()

        loss = self.lossFn(Q_expect, Q_target)

        loss.backward()


        self.optimizer.step()

        self.targetReplace()
        self.step_loss.append(loss.item())
        if self.step % 4 == 0:
            t = np.array(self.step_loss)
            mean = np.mean(self.step_loss)
            print('step: %d, loss: %.10f'%(self.step / 4, mean))
            self.loss_array.append(mean)
            self.step_loss = []
        self.step += 1

    def targetReplace(self):
        self.target.load_state_dict(self.net.state_dict())

    def saveModule(self):
        print()
        torch.save(self.target.state_dict(), self.path)

    def predict(self):
        actionSize = self.env.getActionSize()
        target = Net(actionSize)
        target.load_state_dict(torch.load(self.path))

        self.env.render()
        state = self.env.getState()
        state = np.array([state])
        with torch.no_grad():
            q_next = self.net(state)
            _, index = torch.max(q_next, 1)
            action = index.squeeze(0).detach().item()
            next_state, reward, done, info = self.env.step(action)

        return reward, done

        