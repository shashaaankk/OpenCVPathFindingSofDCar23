"""
For corner detection in an RGB image use `extract_lines_and_corner(img)`.

For a usage example scroll down where it says
```
if __name__ == '__main__':
    ...
```
"""

# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def merge_clusters(lines, labels, angle_threshold=0.5):
    c0 = lines[labels == 0]
    c1 = lines[labels == 1]
    top_lines = []
    top_lines.append(c0[0])
    if len(c1) > 0:
        v0 = np.array(c0[0][0][:2] - c0[0][0][2:])
        v1 = np.array(c1[0][0][:2] - c1[0][0][2:])

        print(v0)
        inverse_angle = np.sum(v0 * v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
        print('dotproduct', inverse_angle)
        if np.abs(inverse_angle) < angle_threshold:
            # Extract the two most confident lines
            top_lines.append(c1[0])
    return top_lines

def cluster_lines(lines):
    # Normalize line directions using sine and cosine of the angles
    line_directions = [
        line[0, :2] - line[0, 2:]
        for line in lines 
    ]
    line_directions = [
        (vec[:, None] @ vec[None, :] / np.linalg.norm(vec)**2).flatten()
        for vec in line_directions 
    ]
    print(line_directions)

    for i in range(3):
        # Use KMeans to cluster lines into two directions
        kmeans = KMeans(n_clusters=2, random_state=i).fit(line_directions)

        # Get the labels assigned to each line
        labels = kmeans.labels_

        if len(merge_clusters(lines, labels)) == 2:
            return labels
    
    return labels

def thinning_zhangsuen(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, img = cv2.threshold(image, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

def line_detection(gray_scale, angle_threshold = .5):
    print(img.shape)
    lines = cv2.HoughLinesP(gray_scale,
        rho=1,
        theta=np.pi / 90,
        threshold=100,
        minLineLength=gray_scale.shape[0] / 10,
        maxLineGap=gray_scale.shape[0] / 10
    )
    if lines is None:
        print('[Error] no line detected')
        return None
    # Sort lines based on confidence (line length)
    lines = np.array(sorted(lines, key=lambda x: x[0, 2], reverse=True))

    # plt.imshow(img)
    # 
    # for l in lines:
    #     plt.plot(l[0,::2], l[0,1::2], color='blue')
    # 
    # plt.colorbar()
    # plt.show()
    # lines = detector.detect(gray_scale)
    print(lines)

    labels = cluster_lines(lines)
    print('clusters', labels)

    top_lines = merge_clusters(lines, labels)

    return np.array(top_lines)

def find_intersection_point(point1, point2, point3, point4):
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4

    m1 = (y2 - y1) / (x2 - x1 + 10e-4)
    m2 = (y4 - y3) / (x4 - x3 + 10e-4)

    b1 = y1 - m1 * x1
    b2 = y3 - m2 * x3

    intersection_x = (b2 - b1) / (m1 - m2 + 10e-4)
    intersection_y = m1 * intersection_x + b1

    return intersection_x, intersection_y

def intersect_lines(l1, l2):
    '''
    line represented as (x1, y1, x2, y2)
    where (x1,y1) and (x2, y2) are points on the line.
    '''

    return find_intersection_point(
        l1[:2], l1[2:],
        l2[:2], l2[2:]
    )

def preprocess(image):
    retval, img = cv2.threshold(image[:, :, 1], 170, 255, cv2.THRESH_BINARY)

    # erosion
    n = image.shape[0] // 200
    kernel = np.ones((n, n), np.uint8)
    # Using cv2.erode() method
    #img = cv2.erode(img, kernel, iterations=1)

    # dilation
    # img = cv2.dilate(img, kernel, iterations=2)

    # edge thining
    img = thinning_zhangsuen(img)
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()

    return img

def extract_lines_and_corner(img: np.array):
    '''
    Parameters:
        img: RGB image as np.array (shape N x M x 3)
    
    Returns:

    If a corner is detected:
        a pair of
        lines: 2 x 4 array with end points of the lines
        corner: (2,) the detected corner point (intersection point of the lines)

    If no corner is detected:
        a pair of
        lines: 1 x 4 array with end points of one line
        corner: None
    '''
    img = preprocess(img)
    img = cv2.dilate(img, np.ones((3, 3)))

    lines = line_detection(img)
    if lines is None:
        return None, None
    
    intersection_point = None
    if len(lines) > 1:
        intersection_point = intersect_lines(lines[0, 0], lines[1, 0])
    
    return lines[:, 0, :], intersection_point


if __name__ == '__main__':
    # This is executed if we call `python corner_detection.py` via the command line
    import os

    for img_name in os.listdir('data'):
    # for img_name in ['frameLshape.jpg']:
        # img_name = 'frameLshape3.jpg'

        img = cv2.imread(f'data/{img_name}')
        assert img is not None
        img = img[10:, 10:]

        lines, corner = extract_lines_and_corner(img)

        # plot preprocessed image
        plt.imshow(img)

        # plot lines
        if lines is not None:
            for l in lines:
                plt.plot(l[::2], l[1::2], color='red', label='all')

        # plot corner
        if corner is not None:
            print('corner detected')
            plt.scatter(corner[0], corner[1], s=100, marker='x', color='pink', linewidths=3)
        else:
            print('no corner')
        
        plt.savefig(f'output/lines_and_corner_{img_name}')
        plt.figure()
        # plt.show()
