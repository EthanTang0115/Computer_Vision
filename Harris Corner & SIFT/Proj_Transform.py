#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This program computes the projective transformation that best fits the feature points in two images
# Yichuan Tang
# 600.661 Computer Vision HW2
#
import numpy as np
import random

def proj(matches, feature_coords_1, feature_coords_2):
    """
    matches: list of index matched features in two images
    feature_coords_1: coordinates of features in image 1
    feature_coords_2: coordinates of features in image 2
    """
    # define parameters for RANSAC algorithm
    num_samples = 4 # need at least 4 points to compute projective transformation
    num_iters = 20000 # iterarions in RANSAC, could be changed
    error_tolerance = 3 # threshold to divide inlier/outlier, could be changed
    proj_trans = RANSAC(matches, feature_coords_1, feature_coords_2, num_samples, num_iters, error_tolerance)
    return proj_trans

def RANSAC(matches, feature_coords_1, feature_coords_2, num_samples, num_iters, error_tolerance):
    matches_idx = [i for i in range(len(matches))]
    used = []
    max_inlier = 0
    proj_best = np.zeros((3,3))
    for i in range(num_iters):
        # local variable of inliers computed from each candidate transformation
        num_inlier = 0
        # generate a random sample of mathced points to compute transformation
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
            A.append([x1, y1, 1, 0, 0, 0, 0, 0, 0])
            A.append([0, 0, 0, x1, y1, 1, 0, 0, 0])
            A.append([0, 0, 0, 0, 0, 0, x1, y1, 1])
            B.append(x2)
            B.append(y2)
            B.append(1)
        proj_estimate = solve_transformation(A,B)
        # compute error for other matches
        error = [] # squared error
        for k in matches_idx:
            if k in samples:
                continue
            (p1, p2) = matches[k]
            (y1, x1) = feature_coords_1[p1]
            (y2, x2) = feature_coords_2[p2]
            # apply transformation estimated
            [x_1_t, y_1_t, one] = np.dot(proj_estimate, [x1, y1, 1])
            error.append(np.sqrt((y_1_t - y2)**2 + (x_1_t - x2)**2))
        # count the number of inliers and find the best transformation
        for l in range(len(error)):
            if error[l] <= error_tolerance:
                num_inlier += 1

        if num_inlier > max_inlier:
            max_inlier = num_inlier
            proj_best = proj_estimate

    return proj_best

def solve_transformation(A, B):
    """
    A: padded matrix includes four coordinates before transformation
    B: vector includes four coordinates after transformation
    """
    t = np.linalg.lstsq(A,B)[0]
    t = np.resize(t, (3, 3))
    return t

