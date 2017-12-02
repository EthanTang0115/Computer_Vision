#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the function used to find strong lines and draw on the image
# Yichuan Tang
#
import numpy as np
from math import sin, cos, pi
import cv2

def p7(image_in, hough_image_in, hough_thresh):
    height, width = hough_image_in.shape[:2]
    peak_hough_image = np.zeros((height,width))
    peak_hough_image[hough_image_in > hough_thresh] = 1
    rho_res = 1
    theta_res = 1

    rho_idxs, theta_idxs = np.nonzero(peak_hough_image)


    line_image_out = cv2.cvtColor(np.uint8(image_in),cv2.COLOR_GRAY2BGR)

    for i in range(len(rho_idxs)):
        rho = rho_idxs[i] * rho_res - height/2
        theta = theta_idxs[i] * theta_res - 90
        a = cos(theta*pi/180)
        b = sin(theta*pi/180)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000* (-b))
        y1 = int(y0 + 1000* (a))
        x2 = int(x0 - 1000* (-b))
        y2 = int(y0 - 1000* (a))
        cv2.line(line_image_out,(x1,y1),(x2,y2),(0,255,0),1)

    return line_image_out
