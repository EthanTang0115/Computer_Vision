#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the python function that detects know objects in new images
# Yichuan Tang
#
import cv2
import numpy as np
from math import atan2, sin, cos

def p4(labels_in, database_in):
    height, width = labels_in.shape[:2]
    labels = np.unique(labels_in)
    num_object_labels = labels.shape[0] - 1
    # Create dictionary
    Attributes = {}
    for i in range (num_object_labels):
        Attributes[i] = {'Area': 0, 'x_Position': 0, 'y_Position': 0, 'x_Second_Moment': 0, 'bilinear_Moment': 0, 'y_Second_Moment': 0, 'a': 0, 'b': 0, 'c': 0, 'Orientation': 0}

    for row in range(height):
        for column in range(width):
            for i in range (num_object_labels):
                if labels_in[row][column] == labels[i+1]:
                    Attributes[i]['Area'] += 1
                    Attributes[i]['y_Position'] += row
                    Attributes[i]['x_Position'] += column
                    Attributes[i]['y_Second_Moment'] += row*row
                    Attributes[i]['x_Second_Moment'] += column*column
                    Attributes[i]['bilinear_Moment'] += -2*row*column
    
    overlays_out = cv2.cvtColor(np.uint8(labels_in * 30),cv2.COLOR_GRAY2BGR)

    for j in range (len(database_in)):
        for i in range (num_object_labels):
            if abs(Attributes[i]['Area'] - database_in[j]['Area']) < 500: # use area as criterion, threshold 700
                Attributes[i]['x_Position'] /= Attributes[i]['Area']
                Attributes[i]['y_Position'] /= Attributes[i]['Area']
                cv2.circle(overlays_out,(Attributes[i]['x_Position'],Attributes[i]['y_Position']),10,(0,255,0),1)

    for j in range (len(database_in)):
        for i in range (num_object_labels):
            if abs(Attributes[i]['Area'] - database_in[j]['Area']) < 500: # use area as criterion, threshold 700
                Attributes[i]['a'] = Attributes[i]['x_Second_Moment'] - Attributes[i]['x_Position']*Attributes[i]['x_Position']*Attributes[i]['Area']
                Attributes[i]['c'] = Attributes[i]['y_Second_Moment'] - Attributes[i]['y_Position']*Attributes[i]['y_Position']*Attributes[i]['Area']
                Attributes[i]['b'] = Attributes[i]['bilinear_Moment'] + 2*Attributes[i]['x_Position']*Attributes[i]['y_Position']*Attributes[i]['Area']
                Attributes[i]['Orientation'] = atan2(Attributes[i]['b'],Attributes[i]['a']-Attributes[i]['c'])/2
                cv2.line(overlays_out,(Attributes[i]['x_Position'] + int(50*cos(-Attributes[i]['Orientation'])),Attributes[i]['y_Position'] + int(50*sin(-Attributes[i]['Orientation']))),(Attributes[i]['x_Position'] - int(50*cos(-Attributes[i]['Orientation'])),Attributes[i]['y_Position']-int(50*sin(-Attributes[i]['Orientation']))),(255,0,0),3)

    database_out = Attributes
    return overlays_out


