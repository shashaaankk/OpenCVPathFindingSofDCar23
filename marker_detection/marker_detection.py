
import cv2
import numpy as np

def detect_all_markers(frame: np.array):
    """Detects various markers

    The detected markers are defined in `cv2.aruco.__dict__['DICT_4X4_1000']`.
    The parking slot marker has id 10.

    Parameters
    ---

    the image as np.array

    Returns
    ---

    a pair of

    * markerCorners: nx4x2 array of corner points where n iy the number of detected markers (typically n=1)
    * markerIds: 2d array of detected ids, typically `None` or `[[10]]`
    """

    marker_type = 'DICT_4X4_1000'
    dict_id = cv2.aruco.__dict__[marker_type]

    print("[INFO] detecting '{}' tags...".format(marker_type))

    # setup aruco detector
    arucoDict = cv2.aruco.getPredefinedDictionary(dict_id)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    # detect markers given in the arucoDict
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

    return markerCorners, markerIds


def detect_marker(frame: np.array):
    """detect the parking slot marker assuming only one is detected
    
    Parameters
    ---

    the image as np.array

    Returns
    ---
    
    None if no marker is detected
    or a 4x2 array containing the corner points of the detected marker
    """
    markerCorners, markerIds = detect_all_markers(frame)

    if markerIds is None:
        return None
    else:
        if len(markerIds) != 1 or len(markerIds[0]) != 1 or markerIds[0, 0] != 10:
            print(f'[Error] Detected marker {np.array(markerIds)}')
            return None

        idxs = np.where(markerIds.flatten() == 10)
        corners = markerCorners[idxs[0][0]]

        return corners
