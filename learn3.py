import time
import sys
import cv2


from memory import Memory
from env import Env

memory = Memory(10000)
env = Env('MsPacman-v0')

env.reset()

for _ in range(1000):
    env.render()
    action = env.getActionSample()
    state = env.getState()
    next_state, reward, done, info = env.step(action)
    memory.add(state, action, reward, next_state, done, info)
    print(done)
    if done:
        print(_)
        break
    #time.sleep(0.01)



env.close()

print(memory.getLen())
#print(memory.data[0])

state, action, reward, next_state, done, info  = memory.getSample(10)
print(state.shape)
print(memory.getLen())

cv2.imwrite('testimg.jpg', state[0][1])