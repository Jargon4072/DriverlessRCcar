import os
import argparse
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

camera = PiCamera()
camera.resolution = (360, 360)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(360, 360))
 
# allow the camera to warmup
time.sleep(0.1)

def create_hue_mask(image, lower_color, upper_color):
    lower = np.array(lower_color, np.uint8)
    upper = np.array(upper_color, np.uint8)
 
    # Create a mask from the colors
    mask = cv2.inRange(image, lower, upper)
    output_image = cv2.bitwise_and(image, image, mask = mask)
    return output_image

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
    image = frame.array
    led1.on()
    # Blur image to make it easier to detect objects
    blur_image = cv2.medianBlur(image, 3)
    

    # Convert to HSV in order to 
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2HSV)
    

    # Get lower red hue
    lower_red_hue = create_hue_mask(hsv_image, [0,50,100], [40,70,70])
    

    # Get higher red hue
    higher_red_hue = create_hue_mask(hsv_image, [40, 50, 100], [80,200,200])    
    

    # Merge the images
    full_image = cv2.addWeighted(lower_red_hue, 1.0, higher_red_hue, 1.0, 0.0)
    

    # Blur the final image to reduce noise from image
    full_image = cv2.GaussianBlur(full_image, (9, 9), 2, 2)
    

    # Convert image to gray in order to find circles in the image
    image_gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    
    
    # Find circles in the image
    circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, 2, 100)

    # If we didn't find circles, the light status is "OFF"
    if circles is None:
        #print ("light is OFF")
        cv2.imshow('final_image',full_image)
        cv2.imshow('image',image)
        cv2.imshow('image_gray',image_gray)

    # If we did find circles, the light is "ON"
    else :
     cv2.imshow('final_image',full_image)
     circles = np.round(circles[0, :]).astype("int")
     for (center_x, center_y, radius) in circles:
       cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 4)
       print("Green light detected")
       led1.off()
     cv2.imshow('image',image)
    rawCapture.truncate(0)
    k=cv2.waitKey(5) & 0xFF
    if k == 27: 
      break

