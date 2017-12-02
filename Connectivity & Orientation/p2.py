#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the python function that inplements sequential labeling algorithm
# Yichuan Tang
#
import cv2
import numpy as np

def Union(label_left,label_up,label_index_array):
    if label_index_array[label_up] > label_index_array[label_left]:
        label_index_array[label_up] = label_index_array[label_left]
    else:
        label_index_array[label_left] = label_index_array[label_up]
    return label_index_array

def Find(label,label_index_array):
    while label_index_array[label] != label:
        label = Find(label_index_array[label],label_index_array)
    return label

def p2(binary_in):
    height, width = binary_in.shape[:2]
    labels_out = np.zeros((height,width), dtype=np.int)
    new_label = 2
    label_array = [0,1]
    label_index_array = [0,1]
# First pass
    for row in range(height):
        for column in range(width):
            current_pixel = binary_in[row][column]
            diagnol_neighbor = binary_in[row-1][column-1]
            up_neighbor = binary_in[row-1][column]
            left_neighbor = binary_in[row][column-1]
            if current_pixel == 1:
                if diagnol_neighbor == 0 and up_neighbor == 0 and left_neighbor == 0:
                    label_array.append(new_label)
                    label_index_array.append(new_label)
                    labels_out[row][column] = new_label
                    new_label += 1
                elif up_neighbor == 1 and left_neighbor == 1:
                    labels_out[row][column] = labels_out[row-1][column]
                    if labels_out[row][column-1] != labels_out[row-1][column]:
                        label_index_array = Union(labels_out[row][column-1],labels_out[row-1][column],label_index_array)
                elif diagnol_neighbor == 1:
                    labels_out[row][column] = labels_out[row-1][column-1]
                elif up_neighbor == 1 and left_neighbor == 0:
                    labels_out[row][column] = labels_out[row-1][column]
                elif up_neighbor == 0 and left_neighbor == 1:
                    labels_out[row][column] = labels_out[row][column-1]

    for i in range(len(label_index_array)):
        label_index_array[i] = Find(i,label_index_array)

    label_unique_array = np.unique(label_index_array)
    label_unique_new_array = [0,1]
    for i in range(2,len(label_unique_array)):
        label_unique_new_array.append(i)

    for i in range(len(label_index_array)):
        for j in range(len(label_unique_array)):
            if label_index_array[i] == label_unique_array[j]:
                label_index_array[i] = label_unique_new_array[j]

# Second pass
    for row in range(height):
        for column in range(width):
            labels_out[row][column] = label_index_array[labels_out[row][column]]

    return labels_out
