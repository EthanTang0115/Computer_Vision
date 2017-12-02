#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This program is written for visualize the matched features
# Yichuan Tang
# 600.661 Computer Vision HW2g
#
import numpy as np
import cv2

def display_matches(feature_coords_1, feature_coords_2, matches, image_1, image_2):
    SbS_image = SbS(image_1, image_2)
    matched_image = line_drawer(SbS_image, feature_coords_1, feature_coords_2, matches)
    return matched_image

def SbS(image_1, image_2):
    SbS_image = np.concatenate((image_1, image_2), axis=1)
    return SbS_image

def line_drawer(SbS_image, feature_coords_1, feature_coords_2, matches):
    num_matches = len(matches)
    shift_offset = SbS_image.shape[1]/2
    for i in range(num_matches):
        line_image = cv2.line(SbS_image, flip(feature_coords_1[matches[i][0]]), flip(shift(feature_coords_2[matches[i][1]], shift_offset)), (0,0,255), 2)
    return line_image

def shift(coord, shift_offset):
    shifted_coord =(coord[0], coord[1] + shift_offset)
    return shifted_coord

def flip(coord):
    flip_coord = (coord[1], coord[0])
    return flip_coord
