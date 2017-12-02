#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the driver script for homework 2
# Yichuan Tang
# 600.661 Computer Vision HW2

import numpy as np
from matplotlib import pyplot as plt
import cv2

from Detect_Features import detect_features
from Match_Features import match_features
from Display_Matches import display_matches
from Affine_Transform import affine
from Proj_Transform import proj
from SIFT import SIFT
from SIFT_match import SIFT_match

def main():
    # read two images to be processed
    path_1 = 'graf1.png' # could be changed to other images
    path_2 = 'graf2.png' # could be changed to other images
    img_1 = cv2.imread(path_1)
    img_2 = cv2.imread(path_2)

    # convert color image into gray scale
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    height, width = img_1_gray.shape[:2]
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # make sure images are of the same size
    if img_2_gray.shape != img_1_gray.shape:
        img_2_gray = cv2.resize(img_2_gray, (width, height))

    if img_2.shape != img_1.shape:
        img_2 = cv2.resize(img_2, (width, height))

    # feature detection
    corner_list_1 = detect_features(img_1_gray, 5, 2.4) # parameters could be changed
    corner_list_2 = detect_features(img_2_gray, 5, 2.8) # parameters could be changed

    # visualize features in Harris corner detection
    for i in range(len(corner_list_1)):
        (y_corner_1, x_corner_1) = corner_list_1[i]
        img_corner_1 = cv2.circle(img_1, (x_corner_1, y_corner_1), 10, (0,255,0), -1)
        
    for j in range(len(corner_list_2)):
        (y_corner_2, x_corner_2) = corner_list_2[j]
        img_corner_2 = cv2.circle(img_2, (x_corner_2, y_corner_2), 10, (0,255,0), -1)

    img_corner_1 = cv2.cvtColor(img_corner_1, cv2.COLOR_BGR2RGB)
    img_corner_2 = cv2.cvtColor(img_corner_2, cv2.COLOR_BGR2RGB)
    #plt.axis('off')
    #plt.imshow(img_corner_1)
    #plt.show()

    # match features
    #print img_1_gray.shape, img_2_gray.shape
    # matches = match_features(corner_list_1, corner_list_2, img_1_gray, img_2_gray)
    # describe features with simplified SIFT
    SIFT_desc_1 = SIFT(corner_list_1, img_1_gray)
    SIFT_desc_2 = SIFT(corner_list_2, img_2_gray)
    # match features using SIFT discriptor
    matches = SIFT_match(corner_list_1, corner_list_2, SIFT_desc_1, SIFT_desc_2)

    # display matches
    img_matched = display_matches(corner_list_1, corner_list_2, matches, img_corner_1, img_corner_2)
    plt.axis('off')
    plt.imshow(img_matched)
    plt.show()

    # compute affine transformation
    affine_trans = affine(matches, corner_list_1, corner_list_2)
    
    # compute projective  transformation
    #proj_trans = proj(matches, corner_list_1, corner_list_2)

    # pad zeros to image borders
    pad_width = 60
    npad = ((pad_width, pad_width), (pad_width, pad_width))
    img_1_pad = np.pad(img_1_gray, pad_width = npad, mode = 'constant', constant_values = 0)
    img_2_pad = np.pad(img_2_gray, pad_width = npad, mode = 'constant', constant_values = 0)
    # wrap image using computed affine transformation
    height, width = img_1_pad.shape[:2]
    img_warp_1 = cv2.warpAffine(img_1_pad, affine_trans[:2], (width, height))
    #img_warp_1 = cv2.warpPerspective(img_1_pad, proj_trans, (width, height))
    #plt.imshow(img_2_pad, cmap='gray', vmin = 0, vmax = 255)
    #plt.show()

    # stitch image together and display
    img_stitch = cv2.addWeighted(img_warp_1, 0.5, img_2_pad, 0.5, 0)
    plt.axis('off')
    plt.imshow(img_stitch, cmap='gray', vmin=0, vmax=255)
    plt.show()

if __name__ == '__main__':
    main()
