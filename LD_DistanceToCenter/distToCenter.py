import cv2

# image path
image_path = r"C:\Users\shash\OneDrive\Desktop\sofDCar\TestImg02.jpg"
lines =[]

frame = cv2.imread(image_path)

# imgGrayScale = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
# retval, img_thresh = cv2.threshold(imgGrayScale, 100, 150, cv2.THRESH_BINARY)

# retval, img_thresh = cv2.threshold(frame[:, :, 1], 150, 255, cv2.THRESH_BINARY)
retval, img_thresh = cv2.threshold(frame[:, :, 1], 225, 255, cv2.THRESH_BINARY)


# Draw grid lines (crosshair) at the center of the image
cv2.line(frame, (0, frame.shape[0] // 2), (frame.shape[1], frame.shape[0] // 2), (255, 0, 0), 1)
cv2.line(frame, (frame.shape[1] // 2, 0), (frame.shape[1] // 2, frame.shape[0]), (255, 0, 0), 1)
resized_original_frame = cv2.resize(frame, (500, 500))
cv2.imshow('Original Frame', resized_original_frame)
cv2.waitKey(0)

cv2.line(img_thresh, (0, img_thresh.shape[0] // 2), (img_thresh.shape[1], img_thresh.shape[0] // 2), (255, 0, 0), 1)
cv2.line(img_thresh, (img_thresh.shape[1] // 2, 0), (img_thresh.shape[1] // 2, img_thresh.shape[0]), (255, 0, 0), 1)
resized_original_img_thresh = cv2.resize(img_thresh, (500, 500))
cv2.imshow('Original Frame', resized_original_img_thresh)
cv2.waitKey(0)

cv2.destroyAllWindows()