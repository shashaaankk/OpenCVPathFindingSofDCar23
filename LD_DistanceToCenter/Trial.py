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

longest_line = None
longest_line_length = 0

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
'''
if lines is not None:
    # Iterate through each detected line
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Length of the line, here we are defining a very large line for visualization
        length = 1000
        # Find the endpoint of the line
        x1 = int(x0 + length * (-b))
        y1 = int(y0 + length * (a))
        x2 = int(x0 - length * (-b))
        y2 = int(y0 - length * (a))
        # Draw the line on the original image for visualization
        cv2.line(cdst, (x1, y1), (x2, y2), (0, 0, 255), 2)

if lines is not None:
    # Iterate through each detected line
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Find the endpoints of the line
        length = 1000  # Defining a very long line for visualization
        x1 = int(x0 + length * (-b))
        y1 = int(y0 + length * (a))
        x2 = int(x0 - length * (-b))
        y2 = int(y0 - length * (a))
        # Calculate the length of the line
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_length > longest_line_length:
            longest_line_length = line_length
            longest_line = (x1, y1, x2, y2)

    # Draw the longest line on the original image for visualization
    x1, y1, x2, y2 = longest_line
    cv2.line(cdst, (x1, y1), (x2, y2), (0, 0, 255), 2)'''

linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 1000, 300)  # Line Height and Gap tp determine the number of lines


def calculate_line_equation(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return m, c


if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
        m, c = calculate_line_equation((l[0], l[1]), (l[2], l[3]))
        print(f"Equation of line {i + 1}: y = {m}x + {c}")

resized_lines = cv2.resize(cdstP, (500, 500))
cv2.line(resized_lines, (resized_lines.shape[1] // 2, 0), (resized_lines.shape[1] // 2, resized_lines.shape[0]),(255, 255, 255), 2)
cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", resized_lines)
cv2.waitKey()

'''
#Line Fitting
# Collect all the line points to fit a line through them
all_points = []
for line in linesP:
    x1, y1, x2, y2 = line[0]
    all_points.append((x1, y1))
    all_points.append((x2, y2))

# Fit a line through the points using numpy's polyfit
# This assumes a linear relationship (y = mx + c)
# Fit a first-degree polynomial (a line) to the data
# This returns m (slope) and c (intercept)
if len(all_points) > 0:
    points_array = np.array(all_points)
    vx, vy, cx, cy = cv2.fitLine(points_array, cv2.DIST_L2, 0, 0.01, 0.01)
    m = vy[0] / vx[0]  # Extracting a single element for slope
    c = cy[0] - m * cx[0]  # Extracting a single element for intercept

    # Construct a line using the slope and intercept
    y1 = int(m * 0 + c)  # Assuming x=0 for the starting point
    y2 = int(m * (resized_lines.shape[1]) + c)  # Assuming x = image width for the end point

    # Create an image and draw the line on it
    line_image = np.zeros_like(resized_lines)
    cv2.line(line_image, (0, y1), (resized_lines.shape[1], y2), (255, 255, 255), 2)

    # Show the constructed line image
    cv2.imshow("Constructed Line Image", line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''