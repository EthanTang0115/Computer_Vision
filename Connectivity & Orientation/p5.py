#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the function implementing an edge magnitude calculator
# Yichuan Tang
#
import numpy as np
from math import pow, sqrt

def boundary_pad (image_in,pad_size):
    height,width = image_in.shape[:2]
    image_padded = np.zeros((height+2*pad_size,width+2*pad_size))
    for i in range(height+2*pad_size):
        for j in range(width+2*pad_size):
            if i >= pad_size and i < height+pad_size and  j >= pad_size and j < width+pad_size :
                image_padded[i][j] = image_in[i-pad_size][j-pad_size]
            elif i < pad_size and j < pad_size:
                image_padded[i][j] = image_in[0][0]
            elif i >= height+pad_size and j < pad_size:
                image_padded[i][j] = image_in[height-1][0]
            elif i < pad_size and j >= width+pad_size:
                image_padded[i][j] = image_in[0][width-1]
            elif i > height+pad_size and j >= width+pad_size:
                image_padded[i][j] = image_in[height-1][width-1]
            elif i < pad_size and j >= pad_size and j < width+pad_size:
                image_padded[i][j] = image_in[0][j-pad_size]
            elif i >= height+pad_size and j >= pad_size and j< width+pad_size:
                image_padded[i][j] = image_in[height-1][j-pad_size]
            elif i >= pad_size and i < height+pad_size and j < pad_size:
                image_padded[i][j] = image_in[i-pad_size][0]
            elif i >= pad_size and i < height+pad_size and j >= width+pad_size:
                image_padded[i][j] = image_in[i-pad_size][width-1]
    return image_padded

def Sobel_operator (image_padded,operator_size,pad_size):
    height,width = image_padded.shape[:2]
    image_convolved = np.zeros((height-2*pad_size,width-2*pad_size))

    if operator_size == 3:
        kernel_x = np.zeros((3,3))
        kernel_y = np.zeros((3,3))
        kernel_x[0][0] = -1
        kernel_x[0][2] = 1 
        kernel_x[1][0] = -2
        kernel_x[1][2] = 2
        kernel_x[2][0] = -1
        kernel_x[2][2] = 1
        kernel_y[0][0] = 1
        kernel_y[0][1] = 2
        kernel_y[0][2] = 1
        kernel_y[2][0] = -1
        kernel_y[2][1] = -2
        kernel_y[2][2] = -1

    if operator_size == 5:
        kernel_x = np.zeros((5,5))
        kernel_y = np.zeros((5,5))
        kernel_x[0][0] = -1
        kernel_x[0][1] = -2
        kernel_x[0][3] = 2
        kernel_x[0][4] = 1
        kernel_x[1][0] = -2
        kernel_x[1][1] = -3
        kernel_x[1][3] = 3
        kernel_x[1][4] = 2
        kernel_x[2][0] = -3
        kernel_x[2][1] = -5
        kernel_x[2][3] = 5
        kernel_x[2][4] = 3
        kernel_x[3][0] = -2
        kernel_x[3][1] = -3
        kernel_x[3][3] = 3
        kernel_x[3][4] = 2
        kernel_x[4][0] = -1
        kernel_x[4][1] = -2
        kernel_x[4][3] = 2
        kernel_x[4][4] = 1
        kernel_y[0][0] = 1
        kernel_y[0][1] = 2
        kernel_y[0][2] = 3
        kernel_y[0][3] = 2
        kernel_y[0][4] = 1
        kernel_y[1][0] = 2
        kernel_y[1][1] = 3
        kernel_y[1][2] = 5
        kernel_y[1][3] = 3
        kernel_y[1][4] = 2
        kernel_y[3][0] = -2
        kernel_y[3][1] = -3
        kernel_y[3][2] = -5
        kernel_y[3][3] = -3
        kernel_y[3][4] = -2
        kernel_y[4][0] = -1
        kernel_y[4][1] = -2
        kernel_y[4][2] = -3
        kernel_y[4][3] = -2
        kernel_y[4][4] = -1

    for i in range(height-2*pad_size):
        for j in range(width-2*pad_size):
            window = image_padded[i+pad_size-(operator_size-1)/2:i+pad_size+(operator_size-1)/2+1,j+pad_size-(operator_size-1)/2:j+pad_size+(operator_size-1)/2+1]
            derivative_x = 0
            derivative_y = 0
            for k in range(operator_size):
                for l in range(operator_size):
                    derivative_x += kernel_x[k][l]*window[k][l]
                    derivative_y += kernel_y[k][l]*window[k][l]
            image_convolved[i][j] += sqrt(pow(derivative_x,2) + pow(derivative_y,2))
    return image_convolved


def p5 (image_in):
    height, width = image_in.shape[:2]
    operator_size = 5 # 3 or 5
    pad_size = (operator_size-1)/2
    image_padded = boundary_pad(image_in, pad_size)
    edge_image_out = Sobel_operator(image_padded,operator_size,pad_size)
    return edge_image_out

