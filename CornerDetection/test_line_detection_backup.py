# Import libraries
import os
import cv2
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

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

    # Draw lines based on their direction
    # result_image = image.copy()
    # for i, line in enumerate(lines):
    #     rho, theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
# 
    #     # Draw lines in different colors based on their cluster label
    #     color = (0, 0, 255) if labels[i] == 0 else (255, 0, 0)
    #     cv2.line(result_image, (x1, y1), (x2, y2), color, 2)
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

def plot_channels(img):
    fig, ax = plt.subplots(2, 3)
    
    for i in range(3):
        mappable = ax[0,i].imshow(img[:,:,i])
        plt.colorbar(mappable)

    hist(ax[1,0], img, 0, 0, 180)
    hist(ax[1,1], img, 1, 0, 255)
    hist(ax[1,2], img, 2, 0, 255)

    # plt.show()
    print(img.shape)

def threshold_hsv(img):
    """not working well"""
    # plt.imshow(img)
    # plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    plot_channels(img)
    print(img.shape)
    img = cv2.inRange(np.array(img), np.array([80., 50, 170.]), np.array([110., 256, 256]))
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    return img[:, :, None]

def threshold_rgb(img):
    plot_channels(img)
    print(img.shape)
    img = cv2.inRange(np.array(img), np.array([0, 240, 0]), np.array([250, 256, 150]))
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    n = img.shape[0] // 7
    kernel = np.ones((n,n),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    return img[:, :, None]


def hist(ax, img, channel, lb, ub):
    hist_channel_g = cv2.calcHist([img], [channel], None, [ub - lb], [lb, ub])
    ax.plot(hist_channel_g, color='green')


def edge_detection(img):
    dst = cv2.Canny(
        img,
        threshold1 = 50,
        threshold2 = 200,
        edges = None,
        apertureSize = 3
    )
    plt.figure()
    plt.imshow(dst)
    plt.colorbar()
    # plt.show()


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

    plt.figure()
    plt.imshow(gray_scale, cmap='gray')
    if lines is None:
        lines = []
        print('No lines!')
    
    for l in lines:
        plt.plot(l[0,::2], l[0,1::2], color='blue', label='all')
    

    for l in top_lines:
        print(l)
        # print(angle(l[0,:2], l[0,2:]))
        plt.plot(l[0,::2], l[0,1::2], color='red', label='representatnts')
    rhos = []

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

    # v1 = l1[:2] - l1[2:]
    # v2 = l2[:2] - l2[2:]
    # 
    # A = np.array([- v1, v2]).T
    # b = l2[2:] - l1[2:]
    # 
    # t = np.linalg.solve(A, b)
    # print(l1[2:] + t[0] * v1)
    # print(l2[2:] + t[1] * v2)
    # 
    # x = l2[2:] + t[1] * v2
    # plt.scatter(x[0], x[1])


def noah_preproc(image):
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

img_name = 'frame1.png'

# img = cv2.imread('../LD_DistanceToCenter/testImg01.jpg')
img = cv2.imread(img_name)
img = img[10:, 10:]
# img = cv2.imread('frameLshape.jpg')
# img = cv2.resize(img, (256, 512))
# img = threshold_rgb(img)
# # edge_detection(img)
# # rgb_gray_scale = np.where(img, np.array([255, 255, 255]), np.array([0, 0, 0]))
# img = thinning_zhangsuen(img)


plt.figure()
plt.imshow(img)

img = noah_preproc(img)
img = cv2.dilate(img, np.ones((3, 3)))
plt.figure()
plt.imshow(img, cmap='gray')

print('after thinning', img.shape, img.dtype)
img_bin = np.zeros_like(img)
img_bin[img > 1] = 255

print('after thresholding', img.shape, img.dtype)

lines = line_detection(img)
intersection_point = intersect_lines(lines[0, 0], lines[1, 0])

print(intersection_point)
plt.scatter(intersection_point[0], intersection_point[1], color='green')
fig.savefig('output')
plt.show()
