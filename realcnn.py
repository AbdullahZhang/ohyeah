import numpy as np
import scipy.io as sio
import torch
from torch import nn
import random
import torch.nn.functional as F
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




batch_size=1810

f= sio.loadmat('-10_90500_10_10_realValue.mat')#读取文件

train_x=torch.tensor(f['data_x'],dtype=torch.float32)[:70000]
test_x=torch.tensor(f['data_x'],dtype=torch.float32)[70000:]
train_y=torch.tensor(f['data_y'].reshape(-1),dtype=torch.int64)[:70000]
test_y=torch.tensor(f['data_y'].reshape(-1),dtype=torch.int64)[70000:]



def train_loader(x,y,batchsize): # 创建迭代器
    lenth=len(x)
    batch_num=lenth//batchsize
    for i in range(batch_num):
        beg=i*batchsize
        end=min((i+1)*batchsize,lenth)
        yield x[beg:end],y[beg:end]


net=nn.Sequential(nn.Conv2d(2,32,2,1,'same'),nn.ReLU(),#9*9
                  nn.Conv2d(32,64,2,1,'same'),nn.ReLU(),#8*8
                  nn.MaxPool2d(2,1),#4*4
                  nn.Conv2d(64,64,3,1,1),
                  nn.MaxPool2d(3,3),#4*4
                  nn.Flatten(),
                  nn.Linear(576,1024),nn.ReLU(),
                  nn.Linear(1024,512),nn.ReLU(),
                  nn.Linear(512,181),
                  nn.Softmax())
loss=nn.CrossEntropyLoss()


def test(test_x,test_y,net):
    test_output = net(test_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    f = open('real_cnn_log.txt', 'a')
    f.write('test accuracy: %.4f' % accuracy)
    f.write('\n')
    print('test accuracy: %.4f' % accuracy)

#net=torch.load('net')
optimer=torch.optim.Adam(net.parameters(),lr=1e-3)
def train(train_loader,train_x,train_y,batch_size,net,loss,optimer,test):
    for epoch in range(500):
        for (a,b) in train_loader(train_x,train_y,batch_size):
            #a=torch.fft.fft2(a)
            out = net(a)
            l = loss(out, b)
            optimer.zero_grad()
            l.backward()
            optimer.step()
            print('epoch ', epoch, ' loss', l)
            #f.write('epoch '+str(epoch)+ ' loss '+str(l))
        test(test_x,test_y,net)
        torch.save(net,'net_realcnn_continuereal')

train(train_loader,train_x,train_y,batch_size,net,loss,optimer,test)

# (a,b)=next(train_loader(train_x,train_y,batch_size))
# print(a.shape)
# print(net(a).shape)