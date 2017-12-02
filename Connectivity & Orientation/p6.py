#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the function implementing hough transform edge detector
# Yichuan Tang
#
import numpy as np
from math import sin, cos, pi
from p1 import p1

def p6(edge_image_in, edge_thresh):
    height, width = edge_image_in.shape[:2]
    edge_image_thresh_out = p1(edge_image_in, edge_thresh)
    
    theta_res = 1
    rho_res = 1

    theta_quantized = np.linspace(-90.0,0.0,np.ceil(90.0/theta_res)+1.0)
    theta_quantized = np.concatenate((theta_quantized, -theta_quantized[len(theta_quantized)-2::-1]))
    diag_len = np.sqrt(width*width+height*height)
    rho_step = np.ceil(diag_len/rho_res)
    nrho = rho_step*2 + 1
    rho_quantized = np.linspace(-rho_step*rho_res,rho_step*rho_res,nrho)

    cos_t = np.cos(theta_quantized*pi/180)
    sin_t = np.sin(theta_quantized*pi/180)

    accumulator = np.zeros((len(rho_quantized),len(theta_quantized)),dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(edge_image_thresh_out)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idxs in range(len(theta_quantized)):
            rho = round(x*cos_t[t_idxs]+y*sin_t[t_idxs]) + diag_len
            accumulator[int(rho),t_idxs] += 1
            
    hough_image_out = accumulator
    
    return edge_image_thresh_out, hough_image_out
