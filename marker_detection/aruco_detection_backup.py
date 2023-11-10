import cv2 as cv

# Simple Aruco Marker Detection

image = cv.imread("4x4_1000-10_laptop_photo.jpg")

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
arucoParams = cv.aruco.DetectorParameters()
arucoDetector = cv.aruco.ArucoDetector(arucoDict, arucoParams)

(corners, ids, rejected) = arucoDetector.detectMarkers(image)

if len(corners) > 0:
	# flatten the ArUco IDs list
	ids = ids.flatten()
	# loop over the detected ArUCo corners
	for (markerCorner, markerID) in zip(corners, ids):
		# extract the marker corners (which are always returned in
		# top-left, top-right, bottom-right, and bottom-left order)
		corners = markerCorner.reshape((4, 2))
		(topLeft, topRight, bottomRight, bottomLeft) = corners
		# convert each of the (x, y)-coordinate pairs to integers
		topRight = (int(topRight[0]), int(topRight[1]))
		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
		topLeft = (int(topLeft[0]), int(topLeft[1]))
		
        # draw the bounding box of the ArUCo detection
		cv.line(image, topLeft, topRight, (0, 255, 0), 2)
		cv.line(image, topRight, bottomRight, (0, 255, 0), 2)
		cv.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
		cv.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
		# compute and draw the center (x, y)-coordinates of the ArUco
		# marker
		cX = int((topLeft[0] + bottomRight[0]) / 2.0)
		cY = int((topLeft[1] + bottomRight[1]) / 2.0)
		cv.circle(image, (cX, cY), 4, (0, 0, 255), -1)
		# draw the ArUco marker ID on the image
		cv.putText(image, str(markerID),
			(topLeft[0], topLeft[1] - 15), cv.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 0), 2)
		print("[INFO] ArUco marker ID: {}".format(markerID))
		# show the output image
		cv.imshow("Image", image)
		cv.waitKey(0)