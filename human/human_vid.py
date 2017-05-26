#code to detect human in live feed
#uses hog feature
#run it as python3 human_vid.py

import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from gpiozero import LED

led1=LED(18)
led2=LED(23)
led3=LED(24)
led4=LED(25)


flag=0
def redetect(image):
		(rects, weights) = hog.detectMultiScale(image, winStride=(8,8), padding=(32,32), scale=1.05)
		print('in redetect')
		if(len(rects)==0):
			led1.off()
			return
		else:
			redetect(image)
    	# draw the original bounding boxes


if __name__ == '__main__':

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    
    # allow the camera to warmup
    time.sleep(0.1)
    led1.off()
        
    #cap=cv2.VideoCapture(0)
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
        image = frame.array
        
        (rects, weights) = hog.detectMultiScale(image, winStride=(8,8), padding=(32,32), scale=1.1)

    	# draw the original bounding boxes
        if(len(rects)>0):
        	for (x, y, w, h) in rects:
          		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
          		print("Human Detected")
          		led1.on()
        else:
         	led1.off()
          #redect(image)         
        #if flag == 0:
         #   break
        #time.sleep(2)   
        cv2.imshow('feed',image)
        rawCapture.truncate(0)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

