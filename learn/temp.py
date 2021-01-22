import random
import os
import statsmodels.api as sm
import numpy as np

import matplotlib.pyplot as plt

a = [1,9]

a = np.array(a)
mean = np.mean(a)
m = np.max(a)
print(mean)
print(m)

path = 'rewards/'

a = os.path.isdir(path)
files = os.listdir(path)

b = int(files[0][0:-4]) + 1
print(b)
b = str(b)+'.jpg'
print(b)

print(os.getcwd())

os.chdir('./learn')

print(os.getcwd())

sample = np.random.uniform(0, 1, 50)

ecdf = sm.distributions.ECDF(sample)

n = 50

x = np.linspace(min(sample), max(sample), num=n)

y = ecdf(x)

plt.figure(figsize=(12, 8))

plt.subplot(221)

plt.plot(x, y)

plt.subplot(222)

plt.plot(x, y)




plt.subplot(223)


plt.plot(x, y)
plt.axhline(y = 0.5, ls=":",c="red")
plt.axvline(x = np.mean(sample), ls="-",c="green")

plt.plot([np.mean(sample)],[ecdf(np.mean(sample))], 'o', color = 'green')
plt.text(np.mean(sample),ecdf(np.mean(sample)), 'P(X <= %.2f) = %.2f'%(np.mean(sample), ecdf(np.mean(sample))))

plt.show()