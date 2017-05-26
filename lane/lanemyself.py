import argparse
from os import listdir
from os.path import isfile,join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

cap=cv2.VideoCapture('road4.mp4')
def color_selection(image):

    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    white_color = cv2.inRange(hls_image, np.uint8([20,200,0]), np.uint8([255,255,255])) ## note that OpenCV uses BGR not RGB
    yellow_color = cv2.inRange(hls_image, np.uint8([10,50,100]), np.uint8([100,255,255]))

    combined_color_images = cv2.bitwise_or(white_color, yellow_color)
    return cv2.bitwise_and(image, image, mask = combined_color_images)
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def vertices():
    vertices = np.array([[(100,imshape[0]),(imshape[1]*.45, imshape[0]*0.6), (imshape[1]*.55, imshape[0]*0.6), (imshape[1],imshape[0])]], dtype=np.int32)
    return vertices

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, color = [0,255,0], thickness = 10):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)

    return lines
def to_keep_index(obs, std=1.5):
    return np.array(abs(obs - np.mean(obs)) < std*np.std(obs))


def avg_lines(lines):

    neg = np.empty([1,3])
    pos = np.empty([1,3])

    ## calculate slopes for each line to identify the positive and negative lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            line_length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0 and line_length > 10:
                neg = np.append(neg,np.array([[slope, intercept, line_length]]),axis = 0)
            elif slope > 0 and line_length > 10:
                pos = np.append(pos,np.array([[slope, intercept, line_length]]),axis = 0)

    ## just keep the observations with slopes with 1.5 std dev
    neg = neg[to_keep_index(neg[:,0])]
    pos = pos[to_keep_index(pos[:,0])]

    ## weighted average of the slopes and intercepts based on the length of the line segment
    neg_lines = np.dot(neg[1:,2],neg[1:,:2])/np.sum(neg[1:,2]) if len(neg[1:,2]) > 0 else None
    pos_lines = np.dot(pos[1:,2],pos[1:,:2])/np.sum(pos[1:,2]) if len(pos[1:,2]) > 0 else None

    return neg_lines, pos_lines
## generate the endpoints of the lane line segments
def gen_endpoints(img, slopes_intercepts):

    imshape = img.shape

    if None not in slopes_intercepts:
        neg_points = [0, np.int(slopes_intercepts[0][0]*0 + slopes_intercepts[0][1]),np.int(imshape[1]*0.45), np.int(slopes_intercepts[0][0]*np.int(imshape[1]*0.45) + slopes_intercepts[0][1])]
        pos_points = [np.int(imshape[1]*0.55), np.int(slopes_intercepts[1][0]*imshape[1]*0.55 + slopes_intercepts[1][1]), imshape[1], np.int(slopes_intercepts[1][0]*imshape[1] + slopes_intercepts[1][1])]
    else:
        return None

    return [neg_points, pos_points]

def gen_lane_lines(img, endpoints, color = [0,255,0], thickness = 7):

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    ## obtain slopes, intercepts, and endpoints of the weighted average line segments
    if endpoints is not None:
        for line in endpoints:

            ## draw lane lines
            cv2.line(line_img, (line[0],line[1]), (line[2],line[3]), color, thickness)

    return line_img

def weighted_img(img, initial_img, α=.9, β=0.95, λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


while(cap.isOpened()):
    ret,frame=cap.read()
    image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #gray=cv2.equalizeHist(image)
    image1=color_selection(frame)
    image2=gaussian_blur(image1,17)
    image3=canny(image2,50,150)
    imshape = image3.shape
    vertices1=vertices()
    image4=region_of_interest(image3,vertices1)
    ## apply hough transformation
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15 # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 25 #minimum number of pixels making up a line
    max_line_gap = 250   # maximum gap in pixels between connectable line segments

    lines = hough_lines(image4, rho, theta, threshold, min_line_len, max_line_gap)
    slopes_intercepts = avg_lines(lines)
    endpoints = gen_endpoints(image, slopes_intercepts)
    lane_lines = gen_lane_lines(image, endpoints=endpoints)
    final_img = weighted_img(lane_lines, frame)

    #cv2.imshow('frame',final_img)
    cv2.imshow('frame',lines)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
