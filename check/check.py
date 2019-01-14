#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:18:10 2019

@author: Nazmul
"""
"""1. Autograd"""
# In[ ]:
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import nltk
import random
import numpy as np
from collections import Counter, OrderedDict
import nltk
from copy import deepcopy
import os
import re
import unicodedata
flatten = lambda l: [item for sublist in l for item in sublist]

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
random.seed(1024)
# In[ ]:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,1)
    def forward(self, x):
        x = self.fc1(x)
        return x
# In[ ]:
net = Net()
print(net)
print(list(net.parameters()))
# In[ ]:
input = Variable(torch.randn(1,1,1), requires_grad = True)
print(input)
# In[ ]:
print(list(net.parameters()))
out = net(input)
print(out)
# In[ ]:
def criterion(out, label):
    return (label - out)**2
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
# In[ ]:
data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]
# In[ ]:
for epoch in range(100):
    for i, data2 in enumerate(data):
        X, Y = iter(data2)
        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), Variable(torch.FloatTensor([Y]), requires_grad=False)
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (i % 10 == 0):
            print("Epoch {} - loss: {}".format(epoch, loss.data[0]))
# In[ ]:
# In[ ]:
# In[ ]:
# In[ ]:
# In[ ]:
x = torch.ones(2,2,requires_grad = True)
x
# In[ ]:
y = x+2
y
# In[ ]:
y.grad_fn
# In[ ]:
z = y*y*3
out = z.mean()
out
# In[ ]:3
a = torch.randn(2,2)
a = ((a*3)/(a-1))
a.requires_grad_(True)
# In[ ]:
out.backward()
# In[ ]:
x.grad
# In[ ]: