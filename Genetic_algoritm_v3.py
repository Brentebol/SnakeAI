# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 21:04:19 2019

@author: brent
"""

import numpy as np
import torch
import random as rnd

from Neural_network import FCNet
from Classes_v2 import Snake, Get_Amount

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def Get_Fitness(score, max_score, min_score):
    x = score - min_score + 1
    x = x / (max_score - min_score + 1)
    return x

def New_Fitness(steps, apples):
    x = float( (steps) + ((2 ** apples) + (apples ** 2) * 500) - (((0.25 * steps) ** 1.3) * (apples**1.3)) )
    #y = (self._frames) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self._frames)**1.3) * (self.score**1.2))
    return x

#inputs the poulation, returns a new population of snakes.
def get_new_population(population, mutation_rate = 0.05, num_elites = 0.05, num_immigrants = 0.1):
    #pop_size = len(population)
    population = population.copy()
    if num_immigrants < 1: #if a percentage is given
        num_immigrants = int(num_immigrants * len(population))
    if num_elites < 1:  #if a percentage is given
        num_elites = int(num_elites * len(population))
    num_children = len(population) - num_immigrants - num_elites
    elites = Get_elites(population, num_elites)
    immigrants = Get_immigrants(population, num_immigrants)
    
    new_population = [0] * num_children
    
    parent_a = Select_Parent(population)
    population[population.index(parent_a)] = 0
    population.remove(0)
    
    parent_b = Select_Parent(population)
    population[population.index(parent_b)] = 0
    population.remove(0)
    
    parent_c = Select_Parent(population)
    
# =============================================================================
#     i = 0
#     while not(parent_a != parent_b != parent_c != parent_a) or i < 1000:
#         i = i + 1
#         if parent_a == parent_b:
#             parent_a = Select_Parent(population)
#         elif parent_c == parent_b:
#             parent_b = Select_Parent(population)
#         elif parent_c == parent_a:
#             parent_c = Select_Parent(population)
# =============================================================================
    
    for i in range(num_children):
        net = FCNet().cuda()
        layer_count = 0
        for layer_a, layer_b, layer_c ,new_layer in zip(parent_a.nn.children(), parent_b.nn.children(),  parent_c.nn.children(), net.children()):
            layer_count += 1
            mask = torch.randint(0,3,layer_a.weight.shape, device = device)
            
            #crossover
            new_weight = torch.zeros(mask.shape).to(device)
            new_weight[mask == 0] = layer_a.weight[mask == 0]
            new_weight[mask == 1] = layer_b.weight[mask == 1]
            new_weight[mask == 2] = layer_c.weight[mask == 2]
            
            #mutation
            mask = torch.rand(layer_a.weight.shape)
            mutation = torch.rand(mask.shape, device = device)
            mutation = (mutation * 2 - 1) * 0.25 #make random between -1, 1, then scale down
            new_weight[mask < mutation_rate] = new_weight[mask < mutation_rate] + mutation[mask < mutation_rate]
            
            new_layer.weight = torch.nn.Parameter(new_weight)
            #new_layer = new_weight
        new_population[i] = Snake(net = net)
        
    new_population = new_population + elites + immigrants
    parent_fit = [parent_a.fit, parent_b.fit, parent_c.fit]
    return new_population, parent_fit


#takes in the population and outputs a potential parent
def Select_Parent(population):
    max_score = 0
    min_score = 0
    for member in population:
        max_score = max(max_score, member.fit)
        min_score = min(min_score, member.fit)
    for member in population:
        member.selection_fit = Get_Fitness(member.fit, max_score, min_score)
    
    for i in range(10000):        
        parent = rnd.choice(population[int(len(population)/2):])
        rnd_var = rnd.random()
        if rnd_var < parent.selection_fit ** 10:
            return parent
            break
        
def Get_elites(population, num_elites):
    population = np.array(population)
    fit = []
    elites = []
    for snake in population:
        fit.append(snake.fit)
    idx = np.array(fit).argsort()[-num_elites:][::-1]
    for i in idx:
        elites.append(Snake(net = population[i].nn))
    return list(elites)
        
def Get_immigrants(population, num_immigrants):
    return Get_Amount(Snake(), num_immigrants)
        
        
        
        