#!/usr/bin/env python
from  __future__  import print_function
import numpy as np
import cv2
import sys
from glob import glob
import itertools as it
import time
import pickle
import math
import imutils
from imutils.video import FPS
from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray

from picamera import PiCamera
from gpiozero import LED

led1=LED(18)
led2=LED(23)
led3=LED(24)
led4=LED(25)
face1 = cv2.CascadeClassifier('/home/pi/Desktop/miniProj/miniproj_resouce/stopSign_detection/stop_sign.xml')
#rawCapture = PiRGBArray(camera, size=(640, 480)) 
# allow the camera to warmup
#time.sleep(0.1)

#video_capture = cv2.VideoCapture(0)
#video_capture.set(cv2.CV_CAP_PROP_FPS, 60)
overlay = cv2.imread("overlay.png", -1)
dist_pickle = pickle.load( open( "calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

#array=[]
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_overlay(img, rect):
    x1, y1, x2, y2 = rect 
    y=y2-y1 + 40
    x=x2-x1 + 40
    small = cv2.resize(overlay, (x, y))

    x_offset = x1 - 10
    y_offset = y1 - 10

    for c in range(0,3):
        img[y_offset:y_offset + small.shape[0], x_offset:x_offset+ small.shape[1], c] = small[:,:,c] * (small[:,:,3]/255.0) + img[y_offset:y_offset+small.shape[0], x_offset:x_offset+small.shape[1], c] * (1.0 - small[:,:,3]/255.0)

def draw_rects(img, rects, color):
	
	v1=0
	for x1, y1, x2, y2 in rects:
		cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
		v1=y1+y2-5
	return v1

def calculate(v, h, x_shift, image):
		alpha = 5.0 * math.pi / 180
		v0 = mtx[1][2]
		ay = mtx[1][1]
		# compute and return the distance from the target point to the camera
		d = h / math.tan(alpha + math.atan((v - v0) /ay))
		if d > 0:
		 cv2.putText(image, "%.1fcm" % d,(image.shape[1]-x_shift -100, image.shape[0]-100 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		return d

ret = True
while(1):
    # Capture frame-by-frame
		#time.sleep(0.5)
		image=vs.read()
		
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)

		found = detect(gray, face1)
		v_param1 = draw_rects(image, found, (0, 255, 0))
		# object detection
        #v_param1 = self.obj_detection.detect(self.stop_cascade, gray, image)
        #v_param2 = self.obj_detection.detect(self.light_cascade, gray, image)
		# distance measuremen
		d1=0
		if v_param1 > 0:
			d1 = calculate(v_param1,8, 300, image)
			#d2 = self.d_to_camera.calculate(v_param2, self.h2, 100, image)
			#self.d_stop_sign = d1
			#self.d_light = d2

		print(d1)
		print(mtx)
		print(dist)
		#for area in array:
		#	print (area)
		# Display the resulting frame
		cv2.imshow('result', image)
		fps.update()
		cv2.waitKey(1)
		# update the FPS counter
		
 
# stop the timer and display FPS information
fps.stop()

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
vs.stop()
