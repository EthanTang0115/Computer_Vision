#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This program implements a simplified version of SIFT feature descriptor
# Yichuan Tang
# 600.661 Computer Vision HW2

import numpy as np
import cv2
from Harris_Corner import Gaussian_derivative
import math

def SIFT(feature_coords, image):
    """
    Input:
    feature_coords: coordinates of features in the image generated using Harris corner detection
    image: the image to describe its features
    """
    # make an dicrtionary to contain feature descriptor for each feature
    descriptor = dict()
    # compute Gaussian gradients
    dy, dx = Gaussian_derivative(image)
    # compute magnitude and orientation of the gradient vector
    mag = np.sqrt(dy**2 + dx**2)
    ori = np.arctan2(dy, dx)
    # quantize the domain of histogram of gradient from 0 to 2*pi
    HoG_domain = [i * 2 * math.pi / 8.0 for i in range(8)]
    # decribe the feature
    height, width = image.shape[:2]
    boundary_offset = 20
    for (i,j) in feature_coords:
        if i < boundary_offset or i > height-(boundary_offset) or j < boundary_offset or j > width-(boundary_offset):
            continue
        window_ori = ori[i-20:i+21, j-20:j+21]
        window_mag = mag[i-20:i+21, j-20:j+21]
        # divide the 41 by 41 window into 16 blocks
        blocks = [k*10 for k in range(4)]
        # create a sift descriptor list for this feature
        sift = []

        for b1 in blocks:
            for b2 in blocks:
                block_ori = window_ori[b1:b1+10, b2:b2+10]
                block_mag = window_mag[b1:b1+10, b2:b2+10]
                # for each block create a histogram
                HoG = [0] * 8

                for y in range(10):
                    for x in range(10):
                        point_ori = block_ori[y][x]
                        point_mag = block_mag[y][x]
                        # least difference to locate vote in histogram
                        vote_idx = np.argmin(abs(HoG_domain - point_ori))
                        # weight the vote with magnitude
                        HoG[vote_idx] += point_mag
                # concatenate HoG of this block into sift feature discriptor list
                sift += HoG

        descriptor[(i,j)] = sift

    # normalize descriptor to account for lighting differences
    for feature_coord, sift in descriptor.items():
        # normalize to unite length
        sift_norm = np.array(sift) / sum(sift)
        # threshold value of each bin, no bin is larger than 0.2
        sift_thresh = [(s if s < 0.2 else 0.2) for s in sift_norm]
        # re-normalize
        sift_thresh_norm = np.array(sift_thresh) / sum(sift_thresh)
        descriptor[feature_coord] = sift_thresh_norm

    return descriptor
