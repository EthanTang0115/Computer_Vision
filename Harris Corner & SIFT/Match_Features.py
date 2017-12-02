#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This program implements cross-correlation feature matching algorithmT
# Yichuan Tang
# 600.661 Computer Vision HW2
#
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def match_features_single_direction (feature_coords1, feature_coords2, image1, image2):
    # inputs are two featrue point lists and two images whose features need to be matched
    # define the list to store matched feature indeces
    matches = []
    # define patch size
    patch_radius = 8
    N = 4.0*(patch_radius+1)*(patch_radius+1)

    # go through all features in image 1 and find corresponding feature in image 2
    num_feature_1 = len(feature_coords1)
    num_feature_2 = len(feature_coords2)
    for i in range(num_feature_1):
        feature_1 = feature_coords1[i]
        row_1 = feature_1[0]
        col_1 = feature_1[1]
        patch_1 = image1[row_1-patch_radius:row_1+patch_radius+1, col_1-patch_radius:col_1+patch_radius+1]
        if patch_complete(patch_1,patch_radius) == 1:
            # normalize the patch
            patch_1_n = normalize_patch(patch_1)
            # cretae variable for index of feature with maximum correlation with patch1
            correlation_max = 0
            match = 0
            for j in range(num_feature_2):
                feature_2 = feature_coords2[j]
                row_2 = feature_2[0]
                col_2 = feature_2[1]
                patch_2 = image2[row_2-patch_radius:row_2+patch_radius+1, col_2-patch_radius:col_2+patch_radius+1]
                if patch_complete(patch_2,patch_radius) == 1:
                    # normalize the patch
                    patch_2_n = normalize_patch(patch_2)
                    correlation = np.sum(patch_1_n*patch_2_n) / N
                    if correlation > correlation_max:
                        correlation_max = correlation
                        match = j
            matches.append((i,match))

    return matches

def normalize_patch (patch):
    # compute normalized patch
    normalized_patch = (patch - np.mean(patch))/np.std(patch)
    return normalized_patch

def match_features (feature_coords1, feature_coords2, image1, image2):
    matches_1 = match_features_single_direction(feature_coords1, feature_coords2, image1, image2)
    matches_2 = match_features_single_direction(feature_coords2, feature_coords1, image2, image1)
    matches_2 = [(s[1], s[0]) for s in matches_2]
    matches = list(set(matches_1) & set(matches_2))
    return matches

def patch_complete(patch,patch_radius):
    patch_size = 2*patch_radius+1
    complete = 1
    height, width = patch.shape[:2]
    if height != patch_size or width != patch_size:
        complete = 0
    return complete





