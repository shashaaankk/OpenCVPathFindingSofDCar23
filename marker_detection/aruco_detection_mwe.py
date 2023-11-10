
import cv2


marker_type = 'DICT_4X4_1000'
dict_id = cv2.aruco.__dict__[marker_type]

print("[INFO] detecting '{}' tags...".format(marker_type))

# setup aruco detector
arucoDict = cv2.aruco.getPredefinedDictionary(dict_id)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

# read image
frame = cv2.imread('4x4_1000-10_phone_photo.jpg')

# detect markers given in the arucoDict
markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

print(
    markerCorners,
    markerIds,
    rejectedCandidates
)
