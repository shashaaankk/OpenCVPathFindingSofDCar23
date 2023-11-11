# Import libraries
import os
import cv2
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import sklearn

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


def angle(x, y):
    # return np.arccos(
    return ((x[0] * y[0] + x[1] * y[1])
        / (np.sqrt(x[0]**2 + y[0]**2) * np.sqrt(x[1]**2 + y[1]**2)))
    # )

def line_detection(gray_scale):
    print(img.shape)
    lines = cv2.HoughLinesP(gray_scale,
        rho=1,
        theta=np.pi / 90,
        threshold=100,
        minLineLength=gray_scale.shape[0] / 10,
        maxLineGap=gray_scale.shape[0] / 10
    )
    # lines = detector.detect(gray_scale)
    print(lines)
    # Sort lines based on confidence (line length)
    lines = sorted(lines, key=lambda x: x[0, 2], reverse=True)

    # Extract the two most confident lines
    top_lines = lines[:2]
    plt.figure()
    plt.imshow(gray_scale, cmap='gray')
    if lines is None:
        lines = []
        print('No lines!')
    for l in top_lines:
        print(l)
        print(angle(l[0,:2], l[0,2:]))
        plt.plot(l[0,::2], l[0,1::2], color='red')
    rhos = []

img = cv2.imread('frame1.png')
img = cv2.imread('frameLshape.jpg')
img = threshold_rgb(img)
# edge_detection(img)
# rgb_gray_scale = np.where(img, np.array([255, 255, 255]), np.array([0, 0, 0]))
img = thinning_zhangsuen(img)

plt.figure()
plt.imshow(img)

print('after thinning', img.shape, img.dtype)
img_bin = np.zeros_like(img)
img_bin[img > 1] = 255

print('after thresholding', img.shape, img.dtype)

line_detection(img)

plt.show()
