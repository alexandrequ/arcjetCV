# import the necessary packages
import numpy as np
import argparse
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
ap.add_argument("-r", "--radius", type = int,
	help = "radius of Gaussian blur; must be odd")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread("pika_2.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (11,11), 0)
# perform a naive attempt to find the (x, y) coordinates of
# the area of the image with the largest intensity value
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

# display the results of the naive attempt
cv2.circle(image, maxLoc, 15, (0, 0, 255), 4)
cv2.imshow("Naive", image)
cv2.imwrite("flowdir.png", image)

widthImg = image.shape[1]
widthLoc = maxLoc[1]

fluxLoc = widthLoc/widthImg

if fluxLoc > 0.5:
	fluxDirection = "left"
elif fluxLoc < 0.5:
	fluxDirection = "right"

print(fluxDirection)


cv2.waitKey(0)
