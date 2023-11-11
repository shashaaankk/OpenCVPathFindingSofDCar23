import cv2

# image path
image_path = r"C:\Users\shash\OneDrive\Desktop\sofDCar\TestImg02.jpg"

frame = cv2.imread(image_path)

retval, img_thresh = cv2.threshold(frame[:, :, 1], 225, 255, cv2.THRESH_BINARY)

# Create a blank image to draw the crosshair grid
crosshair = img_thresh.copy()
# edges
edges = cv2.Canny(crosshair, 100, 200)
# contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    # Calculate contour length
    length = cv2.arcLength(contour, closed=True)

    # Calculate moments to find centroid (center)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Calculate deviation from the center of the image
        img_center = (crosshair.shape[1] // 2, crosshair.shape[0] // 2)  # Center of the image
        deviation = cX - img_center[0]  # Deviation from the center
print(deviation)

# Draw contours and display the image with the detected contours and center
cv2.drawContours(crosshair, contours, -1, (0, 255, 0), 2)  # Draw contours on the image
cv2.circle(crosshair, (cX, cY), 5, (255, 0, 0), -1)  # Mark the center on the image
cv2.line(crosshair, (cX, cY), img_center, (0, 0, 255), 2)  # Draw a line from contour center to image center

# Draw grid lines (crosshair) at the center of the image
cv2.line(crosshair, (10, crosshair.shape[0] // 2), (crosshair.shape[1], crosshair.shape[0] // 2), (255, 255, 255), 1)
cv2.line(crosshair, (crosshair.shape[1] // 2, 0), (crosshair.shape[1] // 2, crosshair.shape[0]), (255, 255, 255), 1)

# Resize the image for display
resized_result_contours = cv2.resize(crosshair, (500, 500))

cv2.imshow('Detected Contours', resized_result_contours)
cv2.waitKey(0)  # Display the result until a key is pressed

# Draw grid lines (crosshair) at the center of the image on the blank image
cv2.line(crosshair, (0, crosshair.shape[0] // 2), (crosshair.shape[1], crosshair.shape[0] // 2), (0, 0, 0), 1)
cv2.line(crosshair, (crosshair.shape[1] // 2, 0), (crosshair.shape[1] // 2, crosshair.shape[0]), (0, 0, 0), 1)

# Display the crosshair grid image
resized_crosshair = cv2.resize(crosshair, (500, 500))
cv2.imshow('Crosshair Grid', resized_crosshair)
cv2.waitKey(0)

cv2.destroyAllWindows()
