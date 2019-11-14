# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:42:53 2019

@author: brent

@todo
azonderlijk muren testen.


notes:
probeer inputs naar u,d,l,r te doen. 4 ipv 8
"""

#%% Global parameters
global field_size
#genetic algortihm paratmers
ga_population = 500
score_set = (2, -2.5) #ratio between score and length of live
score_for_apple = 30
score_for_dying = -30
#game paramters
steps_left = 300
show = 1
field_size = (20,20)
timestep = 0.1 #in seconds
num_immigrants = 0.1 #either integer or % smaller than 1
num_elites = 0.05 #either integer or % smaller than 1


from Neural_network import *
from Classes_v2 import *
from Screen import *
from Genetic_algoritm_v3 import *

try:
    if first_run == False:
        pass
except:
    #%% import modules and libraries
    print("Importing modules for first time.")
    import numpy as np
    import time as time
    import ctypes
    import pygame as pg
    import torch
    import copy
    import matplotlib.pyplot as plt
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda:0' if use_gpu else 'cpu')
    

    
    #%% Defining variables
    global reso
    reso = (500,500)
    
    #Game variables
    count = 0
    pg.display.set_caption('Showing snake ') 
    score = 0
    scores = []
    apple_max = []
    metrics = []
    apple_metrics = []
    
     
    ## Neural Network shizzle
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda:0' if use_gpu else 'cpu')
    
    #Genetic Algoritm parameters
    generation = 0
    ga_dead_snakes = [0] * ga_population
    ga_snakes = Get_Amount(Snake(), ga_population)
    #ga_arrows = Get_Amount(Arrows_keys(), ga_population)

#%%#  declare some objects as function
show = min(show, ga_population)
display = screen(reso, field_size)
surface = display.show()
running = True
main_loop = True   
debug = False

response_time = timestep #change to 0.02 for more viewable result | change to 'timestep' for next level iterations


#%% main loop for genetic algoritm and generation
while main_loop == True:
    generation += 1
    apple_count = 0
    count_over = 0
    mean_score = []
    max_steps = 0
    apple_average = []

    #%% Main Game loop, higher frequency for user input
    while running == True:
        time.sleep(response_time)
        pg.event.pump()
        count += 1
        
        tst = pg.key.get_pressed()
        
        if tst[pg.K_ESCAPE]:
            display.close()
            running = False
            main_loop = False
            debug == True
        
        if tst[pg.K_q]:
            main_loop = False
            debug = True
            
        if tst[pg.K_w]:
            main_loop = False
            first_run = False
        
        #%% main screen update loop
        if count % int(timestep/response_time) == 0: 
            #move and update snakes
            for i, member in enumerate(ga_snakes):
                if member.is_alive:
                    ga_input, member.vision_grid = field_of_view_4(member, field_size)
                    ga_input = ga_input + list(member.dir_vector)
                    ga_input = torch.Tensor(ga_input).view(-1, len(ga_input)).to(device)
                    ga_output = member.nn(ga_input)
                    ga_idx = int(torch.argmax(ga_output))
                    member.dir_vector = np.eye(4)[ga_idx]
             
            #update window    
            display.RedrawWindow(surface, ga_snakes[-show:])
            display.RedrawText(surface, generation, len(ga_snakes), apple_count)
            if show == 1:
                display.draw_vision(surface, ga_snakes[-1])
            display.Draw()  
                #print(ga_output)
                #print(ga_snakes[i].Xpos, ga_snakes[i].Ypos)
            
            #multi Snakes      
            for member in ga_snakes:
                if member.is_alive:
                    member.move()
                    member.steps_left -= 1
                    member.steps_taken += 1
# =============================================================================
#                 #give score to snake
#                 headposition = (member.Xpos, member.Ypos)
#                 appleposition = (apple_unit.Xpos, apple_unit.Ypos)
#                 tailposition = member.tail[-1]
#                 if abs( sum(appleposition) - sum(headposition) ) < abs( sum(appleposition) - sum(tailposition) ):
#                     member.score += score_set[0]
#                 else:
#                     member.score += score_set[1]
#                 #print(member.score)
# =============================================================================
                if member.Xpos == member.appleX and member.Ypos == member.appleY:
                    #member.score += 1
                    member.addcube()
                    member.steps_left = steps_left
                    member.new_apple(field_size)
                    
                    member.apple_count += 1
                    member.steps_taken = 0
                    #member.score += score_for_apple
                    apple_count = max(apple_count, member.apple_count)
                    
                    #if member.score > max_score:
                    #    max_score = member.score  
                    
                if member.check_collision() or member.steps_left < 0:
                    member.is_alive = False
                if member.is_alive == False:
                    member.score = New_Fitness(member.steps_taken, member.apple_count) + np.random.randn() * 10**-10
                    mean_score.append(member.score)
                    apple_average.append(member.apple_count)
                    
                    ga_dead_snakes[count_over] = member
                    ga_snakes[ga_snakes.index(member)] = 0
                    ga_snakes.remove(0)
                    #del member
                    
                    
                    count_over += 1
                    if ga_snakes == []:
                        #print("no more snakes")
                        running = False
                        break
                        
                        

                    
    #%% End of main game loop

                    
    if debug == True:
        break
    max_score, min_score, apple_max = 0, 0, 0
    #print(max_score)
    for new_member in ga_dead_snakes:
        max_score = max(max_score, new_member.score)
        min_score = min(min_score, new_member.score)
        max_steps = max(max_steps, new_member.steps_taken)
        #apple_max = max(apple_max, new_member.apple_count)
        
    fit = [] #used for plotting later
    for new_member in ga_dead_snakes:
        new_member.fit = Get_Fitness(new_member.score, max_score, min_score)
        fit.append(new_member.fit)
    
    
    scores.append((max_score/1000, min_score, np.average(mean_score)/100))
    print('gen: '+str(generation), '  max score: ' + str(round(max_score,3)), '  mean score: ' + str(round(np.average(mean_score),3)), '  max steps: '+str(max_steps))
    apple_metrics.append([apple_count, np.average(apple_average)])
    metrics.append([generation, max_score, min_score, np.average(mean_score), apple_count, np.average(apple_average)])     
    
    #prepare GA algoritm
    other_fit = np.sort(fit)
    other_fit = list(other_fit)
    #staying = [0] * 5
    #for i in range(min(5,ga_population)):
    #    staying[i] = np.argmax(fit)
    #    fit.pop(staying[i])
        
    #sneks = copy.deepcopy(np.array(ga_dead_snakes,dtype='object'))
    #sneks = sneks[staying]
    #rnd_var = np.random.randint(0,ga_population,len(sneks))

        
    #genetic algotirm shizzle
    ga_snakes, parent_fit = get_new_population(ga_dead_snakes, num_elites = num_elites, num_immigrants = num_immigrants)
    idx = [other_fit.index(a) for a in parent_fit]
    #for j in range(len(sneks)):
    #    ga_snakes[rnd_var[j]].nn = sneks[j].nn

    #ga_apple = Get_Amount(Apple(), ga_population)
    running = True
    
    #rnd_var = np.random.randint(0,ga_population,num_immigrants)
    #ga_snakes[list(rnd_var)] = Get_Amount(Snake(),len(rnd_var))
    
    #%% plot all
    show_scores(scores, other_fit, idx, parent_fit, apple_metrics)
    
    #print(len(ga_snakes))
    #pass
    #break


#display.close()

