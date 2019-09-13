import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt

#coordinate calculation, it will start from very bottom and will go 3 blocks
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))  # the main equation is (y= mx+b), 
    x1 = int ((y1- intercept)/slope) # intercept=b , slope = m..
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


#func for average the lines
def avg_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)
    return np.array([left_line, right_line])

# This function will help to coverting the RGB image to very simple image so that les computational power requires...
def canny_filter (image):
    # converting the rgb image to gray scale image,
    # because it requres low computationla power.
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # use of Gaussian Blur method to blured the image ,
    # it helps to detect the changes in pixels. smoothing the images
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # canny function, it also does Gaussain filter for us..
    # it use threshold (50-150 is good) and will detect the high changes in pixels.
    canny = cv.Canny(blur, 50, 150)
    #cv.imshow('result', canny)
    #cv.waitKey(0)
    return canny


# here in this function we will identify the required places from a picture.
def region_of_interst (image):
    height = image.shape[0] #the value of Y
    polygons = np.array([
        [(200, height),  (1100, height)  , (550 , 250)]
    ]) #the traingle we need
    mask = np.zeros_like(image) #this func will make the value of pixels zero means black.
    cv.fillPoly(mask, polygons ,255) # now it will take the mask and the polygons then,
    # it will make the values for polygons area pixels is 255. so the site will be white.

    # this fuction below will do the AND operation between canny filtered image and masked image.
    #  AND operation between those will help us to find the region of interest.
    masked_img = cv.bitwise_and(image, mask)
    return masked_img

#This function will take the hough transform  algo and process it here

def display_lines(image, lines):
    line_img = np.zeros_like(image)
    if lines is not None:
        for x1 , y1, x2, y2 in lines:
            cv.line(line_img,(x1, y1), (x2, y2), (0,255,0), 5)
    return line_img



''''' normally reading a image from dataset
img = cv.imread('DataSet/tti.jpg')
#cv.imshow('result', img)
# need to copy the image because
# if main image change then there will be problem in next steps
lane_img = np.copy(img)
# calling canny filter function
canny_img = canny_filter(lane_img)
croped_img = region_of_interst(canny_img)
# Hough transform
lines = cv.HoughLinesP(croped_img , 2 , np.pi/180, 100 , np.array([]), minLineLength= 40 , maxLineGap= 5)
#smoothing and average of lines
average_lines = avg_slope_intercept(lane_img,lines)
# sending the two parameters in display_lines function, copy of main image and the hough transform algo
line_img = display_lines(lane_img, average_lines)
#combining the lines with the main images.
combo_img = cv.addWeighted(lane_img, 0.8, line_img, 1, 1)
# printing the image in matplot, or we can say as with height and width (x,y)
cv.imshow('result', combo_img)
#plt.imshow(canny)
# cv.waitkey(0) it's not nessasary..
cv.waitKey(0)
cv.destroyAllWindows()
#plt.show()
'''''



# working with vedio , concept is same as previous, just a little bit changes

cap = cv.VideoCapture('DataSet/test2.mp4')
while(cap.isOpened()):
    _, frame = cap.read()

    # calling canny filter function
    canny_img = canny_filter(frame)
    croped_img = region_of_interst(canny_img)
    # Hough transform
    lines = cv.HoughLinesP(croped_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    # smoothing and average of lines
    average_lines = avg_slope_intercept(frame, lines)
    # sending the two parameters in display_lines function, copy of main image and the hough transform algo
    line_img = display_lines(frame, average_lines)
    # combining the lines with the main images.
    combo_img = cv.addWeighted(frame, 0.8, line_img, 1, 1)
    # printing the image in matplot, or we can say as with height and width (x,y)
    cv.imshow('result', combo_img)
    if cv.waitKey(0) & 0xFF== ord('q'):
        break


    cap.release()
    cv.destroyAllWindows()

