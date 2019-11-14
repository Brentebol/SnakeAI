# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:18:44 2019

@author: brent
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FCNet(nn.Module):
    def __init__(self, num_input=16, num_output = 4):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(num_input, 8)
        #self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, num_output)
        
        torch.nn.init.uniform_(self.fc1.weight,-1,1)
        #torch.nn.init.uniform_(self.fc2.weight,-1,1)
        torch.nn.init.uniform_(self.fc3.weight,-1,1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc1(x)
        #x = self.fc2(x)
        x = F.softmax(self.fc3(x), dim = 1)
        return x
    
    def mutate(self):
        pass
        

#def Get_FCNet(number):
#    lst = [0] * number
#    for i in range(number):
#        lst[i] = FCNet().__class__().cuda()
#    return lst


    














