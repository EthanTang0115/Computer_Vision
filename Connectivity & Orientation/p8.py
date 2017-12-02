#! /usr/bin/env python
# -*- coding: utf-8 -*-
# This is the algorithm to implement hough line drawing with end-points detection
# Yichuan Tang
#
import numpy as np
from math import sin, cos, pi
import cv2
from operator import itemgetter

def p8(image_in, edge_thresh_image, hough_image_in, hough_thresh):
    height_img, width_img = edge_thresh_image[:2]
    height_hough, width_hough = hough_image_in.shape[:2]
    diag_len = np.sqrt(width_hough*width_hough+height_hough*height_hough)
    peak_hough_image = np.zeros((height_hough,width_hough))
    peak_hough_image[hough_image_in > hough_thresh] = 1
    rho_res = 1
    theta_res = 1

    y_idxs, x_idxs = np.nonzero(edge_thresh_image)
    rho_idxs, theta_idxs = np.nonzero(peak_hough_image)

    line_image_out = cv2.cvtColor(np.uint8(image_in),cv2.COLOR_GRAY2BGR)
    
    list_of_point_lists = []
    for i in range(len(rho_idxs)):
        point_list = []
        for j in range(len(x_idxs)):
            rho = rho_idxs[i]*rho_res - height_hough/2
            theta = theta_idxs[i]*theta_res - 90
            if (x_idxs[j]*cos(theta*pi/180) + y_idxs[j]*sin(theta*pi/180)+diag_len - rho) < 1:
                point_list.append((y_idxs[j],x_idxs[j]))
        list_of_point_lists.append(point_list)

    print len(list_of_point_lists)

    for i in range(len(list_of_point_lists)):
        current_line = Resort(list_of_point_lists[i])
        current_line_end_points = Findendpoints(current_line)
        print current_line_end_points
        cv2.circle(line_image_out,(current_line_end_points[0][1],current_line_end_points[0][0]),10,(55,255,155),3)
        cv2.circle(line_image_out,(current_line_end_points[1][1],current_line_end_points[1][0]),10,(55,155,155),3)

    return line_image_out


def Resort(point_list):
    y_max = max(point_list,key=itemgetter(1))[0]
    y_min = min(point_list,key=itemgetter(1))[0]
    x_max = max(point_list,key=itemgetter(1))[1]
    x_min = min(point_list,key=itemgetter(1))[1]

    x_range = x_max - x_min
    y_range = y_max - y_min
    if y_range > x_range:
        point_list.sort(key=lambda tup: tup[0])
    elif x_range > y_range:
        point_list.sort(key=lambda tup: tup[1])
        
    return point_list

def Findendpoints(current_line):
        end_points = [current_line[0],current_line[len(current_line)-1]]
        return end_points


