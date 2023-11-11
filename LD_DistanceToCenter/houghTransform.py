import cv2
import numpy as np
import math
# image path
image_path = r"C:\Users\shash\OneDrive\Desktop\sofDCar\TestImg02.jpg"

frame = cv2.imread(image_path)

retval, img_thresh = cv2.threshold(frame[:, :, 1], 225, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(img_thresh, 100, 200)

cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cdstP = np.copy(cdst)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

resized_lines = cv2.resize(cdstP, (500, 500))
cv2.line(resized_lines, (resized_lines.shape[1] // 2, 0), (resized_lines.shape[1] // 2, resized_lines.shape[0]), (255, 255, 255), 2)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", resized_lines)

cv2.waitKey()
cv2.destroyAllWindows()

# resized_edges = cv2.resize(edges, (500, 500))
# cv2.imshow('Edges', resized_edges)
# cv2.waitKey()

# # Draw grid lines (crosshair) at the center of the image
# cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (255, 0, 0), 1)
# cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 0, 0), 1)
# resized_original_frame = cv2.resize(frame, (500, 500))
# cv2.imshow('Original Frame', resized_original_frame)
# cv2.waitKey(0)
#
# cv2.line(img_thresh, (0, img_thresh.shape[0] // 2), (img_thresh.shape[1], img_thresh.shape[0] // 2), (255, 0, 0), 1)
# cv2.line(img_thresh, (img_thresh.shape[1] // 2, 0), (img_thresh.shape[1] // 2, img_thresh.shape[0]), (255, 0, 0), 1)
# resized_original_img_thresh = cv2.resize(img_thresh, (500, 500))
# cv2.imshow('Original Frame', resized_original_img_thresh)
# cv2.waitKey(0)

