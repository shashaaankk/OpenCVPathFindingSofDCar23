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

    # Use KMeans to cluster lines into two directions
    kmeans = KMeans(n_clusters=2, random_state=0).fit(line_directions)

    # Get the labels assigned to each line
    labels = kmeans.labels_

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

def line_detection(gray_scale):
    print(img.shape)
    lines = cv2.HoughLinesP(gray_scale,
        rho=1,
        theta=np.pi / 90,
        threshold=100,
        minLineLength=gray_scale.shape[0] / 5,
        maxLineGap=gray_scale.shape[0] / 10
    )
    # Sort lines based on confidence (line length)
    lines = np.array(sorted(lines, key=lambda x: x[0, 2], reverse=True))

    # lines = detector.detect(gray_scale)
    print(lines)

    labels = cluster_lines(lines)

    print('clusters', labels)

    c0 = lines[labels == 0]
    c1 = lines[labels == 1]

    # Extract the two most confident lines
    top_lines = [c0[0], c1[0]]

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
    retval, img_thresh = cv2.threshold(image[:, :, 1], 170, 255, cv2.THRESH_BINARY)


    # erosion
    n = image.shape[0] // 10
    kernel = np.ones((n, n), np.uint8) 
    
    # Using cv2.erode() method  
    image_eroded = cv2.erode(img_thresh, kernel)


    #dilation
    img_dilation = cv2.dilate(image_eroded, kernel)


    # edge thining
    thinned_image = thinning_zhangsuen(img_thresh)

    return thinned_image

def extract_lines_and_corner(img: np.array):
    '''
    Parameters:
        img: RGB image as np.array (shape N x M x 3)
    Returns:
        a pair of
        lines: 2 x 4 array with end points of the lines
        corner: (2,) the detected corner point (intersection point of the lines)
    '''
    img = preprocess(img)
    img = cv2.dilate(img, np.ones((3, 3)))

    lines = line_detection(img)
    intersection_point = intersect_lines(lines[0, 0], lines[1, 0])

    return lines[:, 0, :], intersection_point


if __name__ == '__main__':
    # This is executed if we call `python corner_detection.py` via the command line
    img_name = 'frame1.png'

    img = cv2.imread(img_name)
    img = img[10:, 10:]

    lines, corner = extract_lines_and_corner(img)

    # plot preprocessed image
    plt.imshow(img)

    # plot lines
    for l in lines:
        plt.plot(l[::2], l[1::2], color='red', label='all')

    # plot corner
    plt.scatter(corner[0], corner[1], color='green')
    plt.savefig(f'output/lines_and_corner_{img_name}')
    plt.show()
