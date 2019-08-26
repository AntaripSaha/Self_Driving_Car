import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


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


# normally reading a image from dataset
img = cv.imread('E:\Thesis\DataSet\Desertroad.jpg')

# need to copy the image because
# if main image change then there will be problem in next steps
lane_img = np.copy(img)

# calling canny filter function
canny = canny_filter(lane_img)
# printing the image in matplot, or we can say as with height and width (x,y)
#cv.imshow('result',canny)

plt.imshow(canny)

# cv.waitkey(0) it's not nessasary..
#cv.waitKey(0)
plt.show()

