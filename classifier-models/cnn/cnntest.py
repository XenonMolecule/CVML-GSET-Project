#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 13:57:22 2018

@author: erikatan
"""

import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
# prepare the test image by converting its resolution to 64 x 64


imagesFolderPath = '/Users/erikatan/Downloads/images/'
model = load_model('vgg_cnn.h5')


total=0
bottle_tp=0
bottle_fn = 0
can_tp=0
can_fn = 0
cardboard_tp=0
cardboard_fn = 0
container_tp=0
container_fn = 0
cup_tp=0
cup_fn = 0
paper_tp=0
paper_fn = 0
scrap_tp=0
scrap_fn = 0
wrapper_tp=0
wrapper_fn = 0
classes = ['bottle', 'can', 'cardboard', 'container', 'cup', 'paper', 'scrap', 'wrapper']
fps = [0, 0, 0, 0, 0, 0, 0, 0]
#predictions = []

imagesFolderContents = os.listdir(imagesFolderPath)

for imgsubfolder in imagesFolderContents:
    if imgsubfolder != '.DS_Store':
        for pic in os.listdir(imagesFolderPath + imgsubfolder)[:10]:
            if pic != '.DS_Store':
                test_image = image.load_img(imagesFolderPath + imgsubfolder + '/' + pic, target_size=(150,150))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)
                result = model.predict(test_image)
                if result.argmax() == 4:
                    print("predicting cup")
                if result.argmax() == 6:
                    print("predicting scrap")
                if imgsubfolder== 'bottle':
                    z=[[1,0,0,0,0,0,0,0,]]
                    if z==result.tolist():
                        bottle_tp+=1
                    else:
                        bottle_fn += 1
                        litter_class = result.argmax()
                        fps[litter_class] += 1
                elif imgsubfolder== 'can':
                    z=[[0,1,0,0,0,0,0,0,]]
                    if z==result.tolist():
                        can_tp+=1
                    else:
                        can_fn += 1
                        litter_class = result.argmax()
                        fps[litter_class] += 1
                elif imgsubfolder== 'cardboard':
                    z=[[0,0,1,0,0,0,0,0,]]
                    if z==result.tolist():
                        cardboard_tp+=1
                    else:
                        cardboard_fn += 1
                        litter_class = result.argmax()
                        fps[litter_class] += 1
                elif imgsubfolder== 'container':
                    z=[[0,0,0,1,0,0,0,0,]]
                    if z==result.tolist():
                        container_tp+=1
                    else:
                        container_fn += 1
                        litter_class = result.argmax()
                        fps[litter_class] += 1
                elif imgsubfolder== 'cup':
                    z=[[0,0,0,0,1,0,0,0,]]
                    if z==result.tolist():
                        cup_tp+=1
                    else:
                        cup_fn += 1
                        litter_class = result.argmax()
                        fps[litter_class] += 1
                elif imgsubfolder== 'paper':
                    z=[[0,0,0,0,0,1,0,0,]]
                    if z==result.tolist():
                        paper_tp+=1
                    else:
                        paper_fn += 1
                        litter_class = result.argmax()
                        fps[litter_class] += 1
                elif imgsubfolder== 'scrap':
                    z=[[0,0,0,0,0,0,1,0,]]
                    if z==result.tolist():
                        scrap_tp+=1
                    else:
                        scrap_fn += 1
                        litter_class = result.argmax()
                        fps[litter_class] += 1
                elif imgsubfolder== 'wrapper':
                    z=[[0,0,0,0,0,0,0,1,]]
                    if z==result.tolist():
                        wrapper_tp+=1
                    else:
                        wrapper_fn += 1
                        litter_class = result.argmax()
                        fps[litter_class] += 1
                total +=1

def get_precision(tp, fp):
    if tp + fp != 0:
        return tp / (tp + fp)
    return 0
    
def get_recall(tp, fn):
    if tp + fn != 0:
        return tp/(tp+fn)
    return 0
    
tp = bottle_tp + can_tp + cardboard_tp + container_tp + cup_tp + paper_tp + scrap_tp + wrapper_tp
print("accuracy: " + str(tp/total))

bottle_precision = get_precision(bottle_tp, fps[0])
bottle_recall= get_recall(bottle_tp, bottle_fn)
print("bottle precision: " + str(bottle_precision))
print("bottle recall: " + str(bottle_recall))
try: 
    print("bottle_f1: " + str(2 * bottle_precision * bottle_recall / (bottle_precision + bottle_recall)))
except ZeroDivisionError:
    print("bottle_f1: " + str(0))
    
can_precision = get_precision(can_tp, fps[1])
can_recall= get_recall(can_tp, can_fn)
print("can precision: " + str(can_precision))
print("can recall: " + str(can_recall))
try: 
    print("can_f1: " + str(2 * can_precision * can_recall / (can_precision + can_recall)))
except ZeroDivisionError:
    print("can_f1: " + str(0))
    
cardboard_precision = get_precision(cardboard_tp, fps[2])
cardboard_recall= get_recall(cardboard_tp, cardboard_fn)
print("cardboard precision: " + str(cardboard_precision))
print("cardboard recall: " + str(cardboard_recall))
try: 
    print("cardboard_f1: " + str(2 * cardboard_precision * cardboard_recall / (cardboard_precision + cardboard_recall)))
except ZeroDivisionError:
    print("cardboard_f1: " + str(0))
    
container_precision = get_precision(container_tp, fps[3])
container_recall= get_recall(container_tp, container_fn)
print("container precision: " + str(container_precision))
print("container recall: " + str(container_recall))
try: 
    print("container_f1: " + str(2 * container_precision * container_recall / (container_precision + container_recall)))
except ZeroDivisionError:
    print("container_f1: " + str(0))
    
cup_precision = get_precision(cup_tp, fps[4])
cup_recall= get_recall(cup_tp, cup_fn)
print("cup precision: " + str(cup_precision))
print("cup recall: " + str(cup_recall))
try: 
    print("cup_f1: " + str(2 * cup_precision * cup_recall / (cup_precision + cup_recall)))
except ZeroDivisionError:
    print("cup_f1: " + str(0))
    
paper_precision = get_precision(paper_tp, fps[5])
paper_recall= get_recall(paper_tp, paper_fn)
print("paper precision: " + str(paper_precision))
print("paper recall: " + str(paper_recall))
try:
    print("paper_f1: " + str(2 * paper_precision * paper_recall / (paper_precision + paper_recall)))
except ZeroDivisionError:
    print("paper_f1: " + str(0))
    
scrap_precision = get_precision(scrap_tp, fps[6])
scrap_recall= get_recall(scrap_tp, scrap_fn)
print("scrap precision: " + str(scrap_precision))
print("scrap recall: " + str(scrap_recall))
try: 
    print("scrap_f1: " + str(2 * scrap_precision * scrap_recall / (scrap_precision + scrap_recall)))
except ZeroDivisionError:
    print("scrap_f1: " + str(0))
    
wrapper_precision = get_precision(wrapper_tp, fps[7])
wrapper_recall= get_recall(wrapper_tp, wrapper_fn)
print("wrapper precision: " + str(wrapper_precision))
print("wrapper recall: " + str(wrapper_recall))
try: 
    print("wrapper_f1: " + str(2 * wrapper_precision * wrapper_recall / (wrapper_precision + wrapper_recall)))
except ZeroDivisionError:
    print("wrapper_f1: " + str(0))
        