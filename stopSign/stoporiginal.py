import cv2
import sys
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from gpiozero import LED


led1=LED(18)
led2=LED(23)
led3=LED(24)
led4=LED(25)


def calculate(v, h, x_shift, image):
		alpha = 5.0 * math.pi / 180
		v0 = mtx[1][2]
		ay = mtx[1][1]
		# compute and return the distance from the target point to the camera
		d = h / math.tan(alpha + math.atan((v - v0) /ay))
		if d > 0:
		 cv2.putText(image, "%.1fcm" % d,(image.shape[1]-x_shift -100, image.shape[0]-100 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		return d

#cascPath = sys.argv[0]
faceCascade = cv2.CascadeClassifier('/home/pi/Desktop/miniProj/miniproj_resouce/stopSign_detection/stop_sign.xml')
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)
flag=0

#def redetect(image):
#	while(1):
		
#	if(flag==1):
#		return
        	
        	
        # draw the original bounding boxes
#video_capture = cv2.VideoCapture(0)
led1.off()
   
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
		image = frame.array
    		
		faces = faceCascade.detectMultiScale(image,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)	
		print(len(faces))
		if(len(faces) > 0):
			for (x, y, w, h) in faces : 
	        		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	        		print("Stop Sign Detected")
	        		led1.on()
	        		#redetect(image)
		else :
			led1.off()
			#flag=1
			#break
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #redetect(image)
        #flag=1
        #break
    #if(flag==1):
    	#break
		cv2.imshow('Video', image)
    
		rawCapture.truncate(0)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


