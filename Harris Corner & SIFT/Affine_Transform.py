#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This program computes the affine transformation that best fit the feature points in two images
# It implements Random Sample Consensus to remove outliner
# Yichuan Tang
# 600.661 Computer Vision HW2
#
import numpy as np
import random

def affine(matches, feature_coords_1, feature_coords_2):
    """
    matches: list of index of matched featrues in two images
    feature_coords_1: coordinates of features in image 1
    feature_coords_2: coordinates of features in image 2
    """
    # deine parameters for RANSAC algorithm
    num_samples = 3 # need at lest 3 points to compute affine transformation
    num_iters = 20000 # iterations in RANSAC, could be changed
    error_tolerance = 3 # threshold to divide inlier/outlier, could be changed
    affine_trans = RANSAC(matches, feature_coords_1, feature_coords_2, num_samples, num_iters, error_tolerance)
    return affine_trans

def RANSAC(matches, feature_coords_1, feature_coords_2, num_samples, num_iters, error_tolerance):
    matches_idx = [i for i in range(len(matches))]
    used = []
    max_inlier = 0
    affine_best = np.zeros((3,3))
    for i in range(num_iters):
        # local variable of inlier computed from each candidate transformation
        num_inlier = 0
        # generate a random sample of matched points to compute transformation
        random.shuffle(matches_idx)
        samples = matches_idx[:num_samples]
        if samples in used:
            i -= 1
            continue
        used.append(samples)
        A = []
        B = []
        # create inputs for least square
        for j in samples:
            (p1, p2) = matches[j]
            # base point
            (y1, x1) = feature_coords_1[p1]
            # transformed point
            (y2, x2) = feature_coords_2[p2]
            A.append([x1, y1, 1, 0, 0, 0])
            A.append([0, 0, 0, x1, y1, 1])
            B.append(x2)
            B.append(y2)
        affine_estimate = solve_transformation(A,B)
        # compute error for other matches
        error = [] # squared error 
        for k in matches_idx:
            if k in samples:
                continue
            (p1, p2) = matches[k]
            (y1, x1) = feature_coords_1[p1]
            (y2, x2) = feature_coords_2[p2]
            # apply transformtion estimated
            [x_1_t, y_1_t, one] = np.dot(affine_estimate, [x1, y1, 1])
            error.append(np.sqrt((y_1_t - y2)**2 + (x_1_t - x2)**2))
        # count the number of inliers and find the best transformation
        for l in range(len(error)):
            if error[l] <= error_tolerance:
                num_inlier += 1

        if num_inlier > max_inlier:
            max_inlier = num_inlier
            affine_best = affine_estimate

    return affine_best

def solve_transformation(A, B):
    """
    A: padded matrix includes three coordinates before transformation
    B: vector includes three coordinates after transformation
    """
    t = np.linalg.lstsq(A,B)[0]
    t = np.resize(t, (2, 3))
    t = np.vstack((t, [0, 0, 1]))
    return t


