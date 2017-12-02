#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the the python function that converts a gray-level image to a binary image using threshold
# Yichuan Tang
#
import cv2
import numpy as np

def p1(gray_in, thresh_val):
    height, width = gray_in.shape[:2]
    binary_out = np.zeros((height,width))
    binary_out[gray_in > thresh_val] = 1
    return binary_out

