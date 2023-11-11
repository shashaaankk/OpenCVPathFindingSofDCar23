import numpy as np
import cv2

lTh = 100
rTh = 200
# Define boundaries for green in BGR color space
lower_green = np.array([0, 100, 0])
upper_green = np.array([80, 255, 80])

image_path = r"C:\Users\shash\OneDrive\Desktop\sofDCar\TestImg02.jpg"  # Replace with your image path

frame = cv2.imread(image_path)

if frame is not None:
    mask = cv2.inRange(frame, lower_green, upper_green)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    # edges
    edges = cv2.Canny(result, lTh, rTh)
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
            img_center = (result.shape[1] // 2, result.shape[0] // 2)  # Center of the image
            deviation = cX - img_center[0]  # Deviation from the center

    # Draw grid lines (crosshair) at the center of the image
    cv2.line(result, (0, result.shape[0] // 2), (result.shape[1], result.shape[0] // 2), (255, 255, 255), 1)
    cv2.line(result, (result.shape[1] // 2, 0), (result.shape[1] // 2, result.shape[0]), (255, 255, 255), 1)

    # Resize the image for display
    resized_result = cv2.resize(result, (500, 500))

    cv2.imshow('Detected Green', resized_result)
    cv2.waitKey(0)  # Display the result until a key is pressed

    # Draw contours and display the image with the detected contours and center
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)  # Draw contours on the image
    cv2.circle(result, (cX, cY), 5, (255, 0, 0), -1)  # Mark the center on the image
    cv2.line(result, (cX, cY), img_center, (0, 0, 255), 2)  # Draw a line from contour center to image center

    # Draw grid lines (crosshair) at the center of the image
    cv2.line(result, (10, result.shape[0] // 2), (result.shape[1], result.shape[0] // 2), (255, 255, 255), 1)
    cv2.line(result, (result.shape[1] // 2, 0), (result.shape[1] // 2, result.shape[0]), (255, 255, 255), 1)

    # Resize the image for display
    resized_result_contours = cv2.resize(result, (500, 500))

    cv2.imshow('Detected Contours', resized_result_contours)
    cv2.waitKey(0)  # Display the result until a key is pressed

    # Draw grid lines (crosshair) at the center of the image
    cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (255, 0, 0), 1)
    cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 0, 0), 1)
    resized_original_frame = cv2.resize(frame, (500, 500))
    cv2.imshow('Original Frame', resized_original_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("Image not loaded. Check the path or file format.")
