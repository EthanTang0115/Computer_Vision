#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the driver program for programming assignment part1
# Yichuan Tang
#
import numpy as np
from matplotlib import pyplot as plt
import cv2
from p1 import p1
from p2 import p2, Union, Find
from p3 import p3
from p4 import p4

# task 1a
img_two_objects = cv2.imread('two_objects.pgm',0)
img_two_objects_binary = p1(img_two_objects, 127)
cv2.imwrite('two_objects_bianry.pgm',img_two_objects_binary)
plt.imshow(img_two_objects_binary*255,'gray')
plt.show()
 
# task 1b
img_two_objects_labeled = p2(img_two_objects_binary)
cv2.imwrite('two_objects_label.pgm',img_two_objects_labeled)
plt.imshow(img_two_objects_labeled*30,'gray')
plt.show()

# task 1c
database,img_two_objects_overlay = p3(img_two_objects_labeled)
print database
plt.imshow(img_two_objects_overlay)
plt.show()

# task 1d
img_many_objects_1 = cv2.imread('many_objects_1.pgm',0)
img_many_objects_2 = cv2.imread('many_objects_2.pgm',0)
img_many_objects_1_binary = p1(img_many_objects_1, 127)
img_many_objects_2_binary = p1(img_many_objects_2, 127)
img_many_objects_1_labeled = p2(img_many_objects_1_binary)
img_many_objects_2_labeled = p2(img_many_objects_2_binary)
img_many_objects_1_overlay = p4(img_many_objects_1_labeled,database)
img_many_objects_2_overlay = p4(img_many_objects_2_labeled,database)
plt.imshow(img_many_objects_1_overlay)
plt.show()
plt.imshow(img_many_objects_2_overlay)
plt.show()

# This is the end of task 1
