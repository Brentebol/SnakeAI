# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:41:54 2019

@author: brent
"""

import pygame as pg

class screen():
    def __init__(self,reso, field_size):
        self.black = (0,0,0)
        self.white = (255,255,255)
        grey = (100,100,100)
        
        self.resolution = reso
        self.unit_length = int(reso[0] / field_size[0])
        self.half_unit_length = int(self.unit_length / 2)
        
        pg.init()
        pg.font.init()
        self.font = pg.font.SysFont("monospace", 20)
        return
    
    def show(self):
        scr = pg.display.set_mode((self.resolution))
        return scr
    
    @staticmethod
    def close():
        pg.display.quit()
    
    @staticmethod
    def Draw():
        pg.display.flip()
    
    def RedrawText(self, disp, generation, amount, score):
        label = self.font.render("Generation: " + str(generation),1,(255,255,255))
        disp.blit(label,(10,6))
        label = self.font.render("Snakes left: " + str(amount),1,(255,255,255))
        disp.blit(label,(10,24))
        label = self.font.render("max_score: " + str(score),1,(255,255,255))
        disp.blit(label,(10,44))
        #print("printed")
    
    def RedrawWindow(self, disp, snake):
        #reset window
        #pg.event.pump()
        disp.fill(self.black)
        size = self.scale(1)
        

        for member in snake:
            #draw the apples
            apple_unit_Xpos = self.scale(member.appleX)
            apple_unit_Ypos = self.scale(member.appleY)
            pg.draw.rect(disp, member.appleColor, (apple_unit_Xpos,  apple_unit_Ypos, size, size), 0)
            
            #draw the snake heads
            head_Xpos = self.scale(member.Xpos)
            head_Ypos = self.scale(member.Ypos)
            pg.draw.rect(disp, member.head_color, (head_Xpos, head_Ypos, size, size), 0)  
            #print("printed")
            
            #draw tails
            for i in range(member.length - 1):
                tail_Xpos = self.scale(member.tail[i][0])
                tail_Ypos = self.scale(member.tail[i][1])
                pg.draw.rect(disp, member.color, (tail_Xpos, tail_Ypos, size, size), 0)  
    
    def scale(self, size):
        scaled_size = self.unit_length * size
        return scaled_size
    
    def draw_vision(self, disp, snake):
        try:
            grid = snake.vision_grid
        except:
            return
        pairs = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]
        size = self.scale(0.2)
        heading = 0
        grid = snake.vision_grid
        for pair in pairs[0::2]:
            x, y = pair

            #for every n square in the looking direction
            for n in range(1, len(grid[0,:]) + 1):
                evaluated_square = (snake.Xpos + n * x, snake.Ypos + n * y)
                X = self.scale(evaluated_square[0])
                Y = self.scale(evaluated_square[1])
                z = grid[heading,n-1]
                color = (0, 0 if z == 1 else 255,  0 if z == 2 else 255)
                #print(color)
                pg.draw.circle(disp, color, (int(X + self.half_unit_length), int(Y + self.half_unit_length)), int(size))
            heading += 1
        return