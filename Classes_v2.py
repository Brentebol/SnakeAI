# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:42:41 2019

@author: brent
"""
import numpy as np
import pygame as pg
import random as rnd
import ctypes
import matplotlib.pyplot as plt
import torch
import os

from Neural_network import FCNet

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda:0' if use_gpu else 'cpu')



class Snake():
    global Ypos
    global Xpos
    
    def __init__(self, net = False, max_steps = 100, field_size = (20,20)):
        self.head_color = (255,255,255)
        self.color = (200,200,200)
        self.length = 3
        self.direction = [1,0]
        self.Xpos = 10
        self.Ypos = 10
        self.score = 0
        self.fit = 0
        self.apple_count = 0
        self.steps_taken = 0
        self.is_alive = True
        self.dir_vector = [0,1,0,0]
        self.total_steps = 0
        #self.vision_grid = False
        

        
        self.steps_left = max_steps
        if net:
            self.nn = net
        else:
            self.nn = FCNet().to(device)
        
        self.tail= []
        for i in range(self.length-1,0,-1):
            self.tail.append((self.Xpos-i,self.Ypos))
            
        #apple charecteristics
        self.appleColor = (255,0,0)
        self.new_apple(field_size)

    def new_apple(self, field_size):
        self.appleX = rnd.randint(1, field_size[0]-2)
        self.appleY = rnd.randint(1, field_size[1]-2)
        if (self.appleX, self.appleY) in self.tail:
            self.new_apple(field_size)
        
        
    def move(self):
        #add new block of tail and delete last block
        self.tail.append((self.Xpos, self.Ypos))
        self.tail.pop(0)
        vector = self.dir_vector
        
        if vector[0] == 1: #up 
            self.direction = [0,-1]
        elif vector[1] == 1: #down
            self.direction = [0,1]
        elif vector[2] == 1: #left
            self.direction = [-1,0] 
        elif vector[3] == 1: #right 
            self.direction = [1, 0]
            
        #move the head in new direction
        self.Xpos += self.direction[0]
        self.Ypos += self.direction[1]
        
    def check_collision(self):
        collision = False
        if self.Xpos >= 20:
            collision = True
        if self.Ypos >= 20:
            collision = True
        if self.Xpos < 0:
            collision = True
        if self.Ypos < 0:
            collision = True
        if (self.Xpos, self.Ypos) in self.tail:
            collision = True
            
        if collision == True:
            self.is_alive = False
        return collision        
            
    
    def addcube(self):
        self.length += 1
        self.tail = [self.tail[0]] + self.tail

class Apple():
    def __init__(self):
        self.color = (255,0,0)
        self.Xpos = rnd.randint(1, 20-2)
        self.Ypos = rnd.randint(1, 20-2)

def Get_Amount(add_class ,number):
    lst = [0] * number
    for i in range(number):
        lst[i] = add_class.__class__()
    return lst

    
def Update_grid(snake, field_size):
    grid = np.zeros((field_size))
    grid[snake.Ypos, snake.Xpos] = int(np.sqrt(sum(np.array(snake.head_color)**2)))
    grid[snake.appleX, snake.appleY] = int(np.sqrt(sum(np.array(snake.appleColor)**2)))
    for x,y in snake.tail:
        grid[y,x] = int(np.sqrt(sum(np.array(snake.color)**2)))
    grid = grid / np.max(grid)
    return grid

def plot_grid(grid):
    #size = len(grid[:,1])
    plt.imshow(grid)

def game_over():
    if ctypes.windll.user32.MessageBoxW(0,  "Try Again?", "Game Over",4) == 6 :
        return Snake(), Apple(), True
    else:
        return Snake(), Apple(), False

def field_of_view_8(snake, field_size, view_range = 10, show = False):
    def calculate(n, view_range):
        x = (n-1)
        x = x * (-1.0/view_range) + 1
        return x
    
    snake_x = snake.Xpos
    snake_y = snake.Ypos
    heading = 0
    neurons = [0]*24
    #check for directions in a clockwise loop starting at 'up'
    pairs = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]
    grid = np.zeros((len(pairs),view_range))
    for pair in pairs:
        x, y = pair

        #for every n square in the looking direction
        for n in range(1, view_range + 1):
            evaluated_square = (snake_x + n * x, snake_y + n * y)
            
            if evaluated_square in snake.tail:
                neurons[heading] = max(neurons[heading], calculate(n, view_range))
                grid[heading,n-1] = 1
            if (np.max(evaluated_square) >= 20) or (np.min(evaluated_square) < 0):
                neurons[heading + 8] = max(neurons[heading + 8], calculate(n, view_range))
                grid[heading,n-1] = 1
            if evaluated_square == (snake.appleX, snake.appleY):
                #neurons[heading + 16] = max(neurons[heading + 16], calculate(n, view_range))
                neurons[heading + 16] = 1
                grid[heading,n-1] = 2
                        
        heading += 1     
    return neurons, grid

def field_of_view_4(snake, field_size, view_range = 10):
    neurons, grid = field_of_view_8(snake, field_size, view_range)
    neurons = neurons[0::2]
    grid = grid[0::2,:]
    return neurons, grid
    

def show_scores(scores, other_fit, idx, parent_fit, metrics):
    plt.clf() #clear the current plot
    plt.subplot(311)
    #scores = np.array(scores)
    #scores[:,0] = np.log10()
    plt.semilogy(scores)
    plt.ylim(0.01, np.max(scores))
    plt.draw()
    
    plt.subplot(312)
    plt.plot(metrics)
    plt.draw()
    
    plt.subplot(313)
    plt.plot(other_fit, 'bo')
    plt.plot(idx, parent_fit, 'rs')
    plt.draw()

def Load_Snakes(Path, num_population):
    new_population = [0] * num_population
    for i in range(num_population):
        net = FCNet()
        net.load_state_dict(torch.load(Path+ "/snake" + str(i) + ".pth"))
        new_population[i] = Snake(net.cuda())
    
    return new_population

def Save_Snakes(population, generation):
    dir_name = "gen"+str(generation)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for i, snake in enumerate(population):
        torch.save(snake.nn, dir_name + "/snake" + str(i) + ".pth")
        
    









