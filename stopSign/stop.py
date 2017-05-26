import cv2
import sys
import math
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import pickle
from gpiozero import LED

led1=LED(18)
led2=LED(23)
led3=LED(24)
led4=LED(25)

#cascPath = sys.argv[0]
stopCascade = cv2.CascadeClassifier('stop_sign.xml')
carCascade=cv2.CascadeClassifier('cars.xml')
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
time.sleep(0.1)
overlay = cv2.imread("overlay.png", -1)
dist_pickle = pickle.load( open( "calibration_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def calculate(v, h, x_shift, image):
		alpha = 5.0 * math.pi / 180
		v0 = mtx[1][2]
		ay = mtx[1][1]
		# compute and return the distance from the target point to the camera
		d = h / math.tan(alpha + math.atan((v - v0) /ay))
		if d > 0:
		 cv2.putText(image, "%.1fcm" % d,(image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		return d


def stop(image, cascade):
	stopface = cascade.detectMultiScale(image,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
	if len(stopface) == 0:
        	return 0
	stopface[:,2:] += stopface[:,:2]
	
	v1=0
	for x1, y1, x2, y2 in stopface:
		cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
		v1=y1+y2-5
	return v1

 #return 0
	
def car(image, cascade):
	carface = cascade.detectMultiScale(image,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
	if len(carface) == 0:
	        return 0
	carface[:,2:] += carface[:,:2]
	v1=0
	for x1, y1, x2, y2 in carface:
		cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
		v1=y1+y2-5
	return v1

led1.off()
  	
#video_capture = cv2.VideoCapture(0)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
  
	#stop sign detection
	ret = stop(image,stopCascade)
	if ret > 0 :
		print("Stop Sign Detected")
		dist=calculate(ret,8, 300, image)
		if dist < 45 :
			led1.on()	
			
	else:
		print("Stop sign not detected")
		led1.off()
    
	
	#car detection
	#ret1=car(image,carCascade)
	#if (ret1>1):
	#	print("car Detected")
	#	dist1=calculate(ret1,8, 300, image)
	#	if(dist1<25):
	#		led1.off()			
	#else:
	#	print("car not detected")

	cv2.imshow('frame',image)   
	rawCapture.truncate(0)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	        break


