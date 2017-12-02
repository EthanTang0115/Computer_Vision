#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This program computes the fundamental matrix
# It implements Random Sample Consensus to remove outliner
# Yichuan Tang
# 600.661 Computer Vision HW3
#
import numpy as np
import random

def RANSAC(matches, feature_coords_1, feature_coords_2, num_samples, num_iters, error_tolerance):
    matches_idx = [i for i in range(len(matches))]
    used = []
    max_inlier = 0
    fund_mat_best = np.zeros((3,3))

    for i in range(num_iters):
        num_inlier = 0
        random.shuffle(matches_idx)
        samples = matches_idx[:num_samples]

        if samples in used:
            i -= 1
            continue

        used.append(samples)

        A = []

        for j in samples:
            (p1, p2) = matches[j]
            (v_l, u_l) = feature_coords_1[p1]
            (v_r, u_r) = feature_coords_2[p2]
            A.append([u_l*u_r, u_l*v_r, u_l, v_l*u_r, v_l*v_r, v_l, u_r, v_r, 1])
        
        # estimate the fundamental matrix
        A = np.array(A)
        U, S, V = np.linalg.svd(A, full_matrices=True)
        o, p = V.shape[:2]
        fund_mat_estimate = np.reshape(V[:,p-1], (3,3))
        
        # test the fundamental matrix estimation with other matches
        error = []

        for k in matches_idx:
            if k in samples:
                continue
            (p1, p2) = matches[k]
            (v_l, u_l) = feature_coords_1[p1]
            (v_r, u_r) = feature_coords_2[p2]
            product = np.dot([u_l, v_l, 1],np.dot(fund_mat_estimate, [u_r, v_r, 1]))
            error.append(np.sqrt(product**2))

        for l in range(len(error)):
            if error[l] <= error_tolerance:
                num_inlier += 1

        if num_inlier > max_inlier:
             max_inlier= num_inlier
             fund_mat_best= fund_mat_estimate

    return fund_mat_best


