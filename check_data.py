#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:38:44 2020

@author: mv01
"""

from PIL import Image, ImageDraw

file = open('/home/mv01/PFLD/data/300W/train_data/list.txt')

count = 0
#line = file.readline().strip().split()
for line in file:
    line = line.strip().split()
    img = Image.open(line[0])
    draw = ImageDraw.Draw(img)
    
    for i in range(68):
        draw.ellipse((float(line[2*i+1])*img.size[0]-1, float(line[2*i+2])*img.size[1]-1, float(line[2*i+1])*img.size[0]+1, float(line[2*i+2])*img.size[1]+1), fill = (0,0,255))
        
    img.show()
    count += 1
    if(count>10): break