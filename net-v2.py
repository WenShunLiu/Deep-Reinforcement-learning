import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np

class Net(nn.Module):
    def __init__(self, actionSize):
        super(Net, self).__init__()
        self.cov1 = nn.Conv2d(4, 16, 8, stride=4)
        self.cov2 = nn.Conv2d(16, 32, 4, stride=2)
        self.fc1 = nn.Linear(2720, 256)
        self.fc2 = nn.Linear(256, actionSize)

    def forward(self, x):
        i, r = self.process(x)
        i = self.cov1(i)
        i = F.leaky_relu(i)
        i = self.cov2(i)
        i = F.leaky_relu(i)
        i = i.view(i.size(0), -1)
        i = torch.cat((i, r), dim=1)
        i = self.fc1(i)
        i = F.leaky_relu(i)
        i = self.fc2(i)
        return i


    def process(self, x):
        batch = x.shape[0]
        rams = []
        imgs = []
        trams = T.Compose([
            T.ToTensor(),
            T.Normalize([0], [1])
        ])
        for i in range(batch):
            ram =  np.array(x[i][0])
            img = x[i][1]
            img = Image.fromarray(img).convert("L")
            img = T.Resize([110, 84])(img)
            img = img.crop((0, 0, 84, 84))
            img = np.stack((img, img, img, img), axis = 2)
            img = trams(img)
            img = img.numpy()
            imgs.append(img)
            ram = (ram - np.mean(ram))/np.std(ram) # 中心化 均值0，方差1
            rams.append(ram)
        imgs = torch.tensor(imgs)
        rams = torch.tensor(rams, dtype=torch.float)
        return imgs, rams


if __name__ == "__main__":
    img = cv2.imread('testimg.jpg')
    print(img)
    print(img.shape)

    im = np.array(Image.fromarray(img).convert('L'))
    print(im)
    print(im.shape)
    cv2.imwrite("a.jpg", im)

    i = cv2.pyrDown(im)
    print(i)
    print(i.shape) # (105*80)
    cv2.imwrite("b.jpg", i)
    h, w = i.shape
    r = i[0:85, 0:w]
    print(r)
    cv2.imwrite("r.jpg", r) #(85*80)
    net = Net()
    print(net)


    img = Image.open('testimg.jpg').convert('L')
    i = T.Resize([110, 84])(img)
    i = i.crop((0, 0, 84, 84))
    i.save("t.jpg")
    