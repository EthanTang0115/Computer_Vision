#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the driver program for programming assignment part2
# Yichuan Tang
#
import numpy as np
from matplotlib import pyplot as plt
import cv2
from p5 import p5
from p6 import p6
from p7 import p7
from p8 import p8

img_simple_1 = cv2.imread('hough_simple_1.pgm',0)
img_simple_2 = cv2.imread('hough_simple_2.pgm',0)
img_complex_1 = cv2.imread('hough_complex_1.pgm',0)

img_simple_1_edge = p5(img_simple_1)
img_simple_2_edge = p5(img_simple_2)
img_complex_1_edge = p5(img_complex_1)
plt.imshow(img_simple_1_edge,'gray')
plt.show()
plt.imshow(img_simple_2_edge,'gray')
plt.show()
plt.imshow(img_complex_1_edge,'gray')
plt.show()

img_simple_1_edge_threshold, img_simple_1_hough = p6(img_simple_1_edge, 180)
img_simple_2_edge_threshold, img_simple_2_hough = p6(img_simple_2_edge, 180)
img_complex_1_edge_threshold, img_complex_1_hough = p6(img_complex_1_edge, 180)
plt.imshow(img_simple_1_edge_threshold,'gray')
plt.show()
plt.imshow(img_simple_2_edge_threshold,'gray')
plt.show()
plt.imshow(img_complex_1_edge_threshold,'gray')
plt.show()

img_simple_1_hough_line = p7(img_simple_1, img_simple_1_hough, 420)
img_simple_2_hough_line = p7(img_simple_2, img_simple_2_hough, 420)
img_complex_1_hough_line = p7(img_complex_1, img_complex_1_hough, 100)
plt.imshow(img_simple_1_hough_line)
plt.show()
plt.imshow(img_simple_2_hough_line)
plt.show()
plt.imshow(img_complex_1_hough_line)
plt.show()

img_complex_1_hough_line_end = p8(img_complex_1, img_complex_1_edge_threshold, img_complex_1_hough, 250)
plt.imshow(img_complex_1_hough_line_end,'gray')
plt.show()
