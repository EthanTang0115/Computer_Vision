#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This program implments matching algorithm for SIFT fetaure descriptorusing ratio test
# Yichuan Tang
# 600.661 Computer Vision HW2

import numpy as np

def SIFT_match(feature_coords_1, feature_coords_2, descriptor_1, descriptor_2):
    """
    Input:
    feature_coords_1: coordinates of features in the first image generated using Harris corner detection
    feature_coords_2: coordinates of features in the second image generated using Harris corner detection
    descriptor_1: SIFT descriptor of features in the first image, a dictionary
    descriptor_2: SIFT descriptor of fetaures in the second image, a dicitionary
    """
    matches = []
    ratio_thresh = 0.6
    match_best = tuple()
    for c1, (y1, x1) in enumerate(feature_coords_1):
        # not all feature points has a descriptor
        if (y1, x1) not in descriptor_1:
            continue
        desc_1 = descriptor_1[(y1, x1)]
        # initialize a large number for L2 distance
        min_L2_dist = float('inf')
        second_min_L2_dist = float('inf')

        for c2, (y2, x2) in enumerate(feature_coords_2):
            if (y2, x2) not in descriptor_2:
                continue
            desc_2 = descriptor_2[(y2, x2)]
            diff = desc_1 - desc_2
            L2_dist = np.sqrt(np.sum(diff ** 2))

            if L2_dist < min_L2_dist:
                min_L2_dist = L2_dist
                best_match = (c1, c2)
            elif L2_dist < second_min_L2_dist:
                second_min_L2_dist2 = L2_dist
        # compute ration of mininum distance over the second minimum distance
        ratio = min_L2_dist / second_min_L2_dist

        if ratio <= ratio_thresh:
            matches.append(best_match)

    return matches
