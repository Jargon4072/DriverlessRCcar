# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def draw_overlay(img, rect1):
    x1, y1, x2, y2 = rect1
    y=y2-y1 + 40
    x=x2-x1 + 40
    small = cv2.resize(overlay, (x, y))

    x_offset = x1 - 10
    y_offset = y1 - 10

    for c in range(0,3):
        img[y_offset:y_offset + small.shape[0], x_offset:x_offset+ small.shape[1], c] = small[:,:,c] * (small[:,:,3]/255.0) + img[y_offset:y_offset+small.shape[0], x_offset:x_offset+small.shape[1], c] * (1.0 - small[:,:,3]/255.0)

def draw_rects(img, rects1, color):
		for x1, y1, x2, y2 in rects1:
			cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)



# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()

# loop over some frames...this time using the threaded stream
while(1):
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	#frame = imutils.resize(frame, width=400)

	image = imutils.resize(frame, width=min(400,frame.shape[1]))
	orig = image.copy()
	

	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
	padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
	cv2.imshow("feed", orig)
	#cv2.imshow("After NMS", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
#vs.stop()
