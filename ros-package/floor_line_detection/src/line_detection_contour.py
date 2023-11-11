
import os
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math

def lineDetectionContour(image):
    """
    The function returns distance
    input: image
    output: distance, directionFlag,
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image (white and black)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours based on their area (largest to smallest)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #print(contours)
    # Extract the largest contour
    largest_contour = contours[0]
    # Find the endpoints of the largest contour
    leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    #print('leftmost point:', leftmost)
    #print('rightmostpoint:', rightmost)
    # get the image shape
    height, width, channels = image.shape
    bottom_middle_point = (width // 2, height)
    slop = (leftmost[1] - rightmost[1])/(leftmost[0] - rightmost[0])
    #print(leftmost[1])
    #print(slop)
    directionFlag = 0;    # if + right adjustment, if - left adjustment
    if slop < 0.0:
        distance = math.sqrt((bottom_middle_point[0] - leftmost[0])**2 + (bottom_middle_point[1] - leftmost[1])**2)
        directionFlag = -1;  # turn left
    else:
        distance = math.sqrt((bottom_middle_point[0] - rightmost[0])**2 + (bottom_middle_point[1] - rightmost[1])**2)
        directionFlag = +1;  # turn right
    #print('disst:', distance)

    return distance, directionFlag
