import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("pika.png")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convert the image in gray
# create a CLAHE object (Arguments are optional).
#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#frame = clahe.apply(gray)
blurred = cv.GaussianBlur(gray, (11, 11), 0) # smoothing (blurring) it to reduce high frequency noise
thresh = cv.threshold(blurred, 250, 255, cv.THRESH_BINARY)[1] # threshold the image to reveal light regions in the
        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
thresh_1 = cv.erode(thresh, None, iterations=2)
thresh_2 = cv.dilate(thresh_1, None, iterations=4)
edges = cv.Canny(thresh_2,200,100)
contours, hierarchy = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        #cv.drawContours(self.frame, self.contours, -1, (0,96,196), 3)
        #cv.imshow("hello", frame)
frame = cv.drawContours(img, contours, -1, (0,96,196), 3)

cv.imwrite('pika_blurred.png', blurred)
cv.imwrite('pika_threshold.png', thresh)
cv.imwrite('pika_erode.png', thresh_1)
cv.imwrite('pika_dilate.png', thresh_2)
cv.imwrite('pika_gradient.png', edges)
cv.imwrite('pika_final.png', frame)
