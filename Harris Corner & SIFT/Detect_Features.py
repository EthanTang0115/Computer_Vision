#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the function that implements Harris corner detection method
# Yichuan Tang
# 600.661 Computer Vision HW2

import cv2
import numpy as np
from nonmaxsuppts import nonmaxsuppts

def detect_features(image, nonmax_radius, corner_thresh_ratio):
    """
    Input:
    image
    nonmax_radius: radius of window for non-maximum suppression
    corner_thresh_ration: threshold of the normalized corner strength
    """
    # define key parameters
    window_size = 9
    offset = (window_size-1)/2
    k = 0.05
    # create a list to store corner coordinates
    corner_list= []
    #create corner strength image
    height, width = image.shape[:2]
    corner_strength = np.zeros((height, width))
    # compute Gaussian gradients
    dy, dx = Gaussian_derivative(image)
    # compute second moments
    Iyy = dy**2
    Iyx = dy*dx
    Ixx = dx**2

    #compute corner strength image
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            #compute second moments
            a_window = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            b_window = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            c_window = Iyx[y-offset:y+offset+1, x-offset:x+offset+1]
            a = a_window.sum()
            b = b_window.sum()
            c = c_window.sum()
            # det equals multiplication of two eigenvalues
            det = (a*b) - c**2
            # trace equals sum of two eigenvalues
            trace = a + b
            # compute r
            r = det - k*(trace**2)
            # fill the corner strength image
            corner_strength[y][x] = r

    # normalize corner strength
    corner_strength = (corner_strength - np.min(corner_strength)) * 10.0 / (np.max(corner_strength - np.min(corner_strength)))

    # apply threshold to find corner points and highlight on the image
    corner_thresh = corner_thresh_ratio * np.mean(corner_strength)
    corner_list = nonmaxsuppts(corner_strength, nonmax_radius, corner_thresh)
    return corner_list


def Gaussian_derivative(image):
    """
    Input: 
    image

    Output:
    gaussian derivative of the image with respect to y and x direction
    """
    height, width = image.shape[:2]
    dy = np.zeros((height, width))
    dx = np.zeros((height, width))
    Gaussian_kernel = cv2.getGaussianKernel(7,3)
    Gaussian_kernel_2D = Gaussian_kernel*Gaussian_kernel.T
    image_blurred = cv2.filter2D(image,-1,Gaussian_kernel)

    #apply sober operator to compute derivatives
    kernel_y = np.zeros((3,3))
    kernel_x = np.zeros((3,3))
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
    
    # compute derivative for each pixel
    for i in range(1,height-1):
        for j in range(1,width-1):
            window = image[i-1:i+2,j-1:j+2]
            derivative_y = kernel_y*window
            derivative_x = kernel_x*window
            dy[i][j] = derivative_y.sum()
            dx[i][j] = derivative_x.sum()
    
    return dy, dx

