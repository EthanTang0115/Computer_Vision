#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the program that implements ORB, finds the fundamental matrix, and draws epipolar lines
# It calls ORB function from scikit-image
# Yichuan Tang
# 600.661 Computer Vision HW3

import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

from Detect_Features import detect_features
from Display_Matches import display_matches
from RANSAC import RANSAC
from SIFT import SIFT
from SIFT_match import SIFT_match
####################utility functions######################
def plot_epipolar_line(SbS, fundamental_matrix, x, color):
# thsi function draws epipolar lines on the SbS image
	m,n = SbS.shape[:2]
	shift_offset = n/2
	F = fundamental_matrix
	line = [(F[0][0]*x[0]+F[1][0]*x[1]+F[2][0]), (F[0][1]*x[0]+F[1][1]*x[1]+F[2][1]), (F[0][2]*x[0]+F[1][2]*x[1]+F[2][2])]
	# epipolar line parameters and values
	t = np.linspace(0, n/2, 100)
	lt = np.array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])
	# take only line points inside the image
	ndx = (lt>=0) & (lt<m)
	plt.plot(t+shift_offset,lt, color=color, linewidth=2)
####################end of functions######################

####################start of the main program######################
# load images
path1 = 'hopkins1.JPG'
path2 = 'hopkins2.JPG'
img_1 = cv2.imread(path1)
img_2 = cv2.imread(path2)

# convert color image into gray scale
img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
height, width = img_1_gray.shape[:2]
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# detect features
corner_list_1 = detect_features(img_1_gray, 5, 1) # parameters could be changed
corner_list_2 = detect_features(img_2_gray, 5, 1) # parameters could be changed
corner_list_1 = corner_list_1[:100]
corner_list_2 = corner_list_2[:100]

# generate feature descriptors
SIFT_desc_1 = SIFT(corner_list_1, img_1_gray)
SIFT_desc_2 = SIFT(corner_list_2, img_2_gray)

# match features
matches = SIFT_match(corner_list_1, corner_list_2, SIFT_desc_1, SIFT_desc_2)

# visualize matches
img_matched = display_matches(corner_list_1, corner_list_2, matches, img_1, img_2)
plt.axis('off')
plt.imshow(img_matched)
plt.show()

# compute fundamental matrix using RANSAC
num_samples = 8
num_iters = 2000
error_tolerance = 10
fundamental_matrix = RANSAC(matches, corner_list_1, corner_list_2, num_samples, num_iters, error_tolerance)
print fundamental_matrix

# draw 8 feature points and epipolar lines
SbS = np.concatenate((img_1, img_2), axis=1)
m,n = SbS.shape[:2]
shift_offset = n/2

# take 8 random feature points
all_inds = [i for i in range(len(matches))]
random.shuffle(all_inds)
inds = all_inds[:8]

# Draw feature points and epipolar lines on SbS image
plt.figure()
plt.imshow(SbS)
color_bank = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
counter = 0
for i in inds:
	(p1, p2) = matches[i]
	(v_l, u_l) = corner_list_1[p1]
	(v_r, u_r) = corner_list_2[p2]
	color = color_bank[counter]
	plt.plot(u_l, v_l, 'o', color=color_bank[counter])
	plt.plot(u_r+shift_offset, v_r, 'o', color=color_bank[counter])
	x = np.array([u_l, v_l, 1])
	plot_epipolar_line(SbS, fundamental_matrix, x, color_bank[counter])
	counter += 1

plt.axis('off')
plt.show()
####################end of the main program######################






