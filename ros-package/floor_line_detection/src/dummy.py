#!/usr/bin/env python

import rospy
import cv2
from geometry_msgs.msg import  Pose
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
#from corner_detection.py  import extract_lines_and_corner
from direction_turn_detector import find_turn_direction
from line_detection_contour import lineDetectionContour
#from marker_detection import detect_marker


bridge = CvBridge()

pub = rospy.Publisher("/line_output", Pose, queue_size=10)

def callback(image):
	
	try:
        	cv_image = bridge.imgmsg_to_cv2(image)	
	except CvBridgeError as e:
		print(e)

	np_image = np.asarray(cv_image)
	turn_flag = find_turn_direction(cv_image)
	#print(turn_flag)
	distance,direction_flag = lineDetectionContour(cv_image)
	
	#print(direction_flag)
	#corners = detect_marker(cv_image)
	#print(corners)
	
	pose_line = Pose()

	pose_line.position.x = direction_flag
	pose_line.position.y = turn_flag
	pose_line.position.z = distance

	#  cv2.imshow("image",np_image)
	#cv2.waitKey(3)
	#print(np_image.shape)
	#np.reshape(npimage,(640,480,3))
	#print(npimage)
	
	#print(type(image))
	pub.publish(pose_line)

def  detector():
	rospy.init_node('detector', anonymous=True)
	# pub = rospy.Publisher("/line_output", Pose, queue_size=10)
	# pose_line = Pose()
	
	rospy.Subscriber("/csi_cam_0/image_raw", Image, callback)
	
	rospy.spin()

if __name__ == '__main__':
	detector()
