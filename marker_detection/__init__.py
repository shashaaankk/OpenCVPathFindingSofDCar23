"""
Usage:

```py
import marker_detection

# produce some image
import cv2
img = cv2.imread('marker_detection/4x4_1000-10_phone_photo.jpg')

# find the parking slot marker
res = marker_detection.detect_marker(img)
print(res)

# output: the corner points of the marker
# [[[ 392. 1113.]
#   [2337.  936.]
#   [2515. 3004.]
#   [ 113. 3004.]]]

# if there are multiple markers (advanced scenario),
# we may need to use the full output
res = marker_detection.detect_all_markers(img)
print(res)

# output:
# ((array([[[ 392., 1113.],
#         [2337.,  936.],
#         [2515., 3004.],
#         [ 113., 3004.]]], dtype=float32),),
#   array([[10]], dtype=int32))
```
"""

from .marker_detection import detect_marker, detect_all_markers
