import cv2
import numpy as np

# Input - tuple values, Output - -1 for left turn and +1 for right turn   
def findMaxValueFromTupleAndReturnIndex(value):
    print(value)
    max_val = max(value)
    cmp = value.index(max_val)
    res = -1 if cmp == 0 else cmp
    print(res)
    return res    
  
# Input - contours of points that fall under green path and step size for controlling the algo, 
# Output - generates output from findMaxValueFromTupleAndReturnIndex method
def detect_turns_direction(contours, step_size=5):
    # Function to detect the direction of turns based on the slope of the line between consecutive points
    val_arr = []
    # Iterate through contours
    for contour in contours:
        # Iterate through points in the contour with a specified step size
        for i in range(0, len(contour) - 2, step_size):
            # Get three consecutive points
            pt1 = contour[i][0]
            pt2 = contour[i + 1][0]
            pt3 = contour[i + 2][0]

            # Calculate vectors between points
            vector1 = pt1 - pt2
            vector2 = pt3 - pt2

            # Calculate the slope of the line between points
            slope = (vector2[1] - vector1[1]) / (vector2[0] - vector1[0] + 1e-5)  # Avoid division by zero

            threshold_slope = 0.5  # Adjust this value based on your requirements

            # Determine the direction of the turn based on the slope
            if slope > threshold_slope:
                print("Right turn ahead")
                val_arr.append(-1)
            elif slope < -threshold_slope:
                print("Left turn ahead")
                val_arr.append(1)
            # else:
            #     val_arr.append(0)    
    value = (val_arr.count(1), val_arr.count(-1))
    return findMaxValueFromTupleAndReturnIndex(value)           

# Load the image
image = cv2.imread('./data/path5.jpeg')

# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV color range for green
green_lower = np.array([40, 40, 40])  # Lower bound for green in HSV
green_upper = np.array([80, 255, 255])  # Upper bound for green in HSV

# Thresholding to extract green-colored regions
mask_green = cv2.inRange(hsv_image, green_lower, green_upper)

# Edge detection using canny edge detection
edges_green = cv2.Canny(mask_green, 50, 150)

# Contour detection
contours_green, _ = cv2.findContours(edges_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by green color
green_contours = [contour for contour in contours_green]

# Detect the direction of turns in green-colored contours
detect_turns_direction(green_contours, step_size=10) 
