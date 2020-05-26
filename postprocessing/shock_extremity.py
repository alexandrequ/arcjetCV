import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from scipy.interpolate import splev, splprep, interp1d

img = cv2.imread('shock_detection/tests/frame_0012_out.png')   # you can read in images with opencv
#img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def shock_extremity(img_mask, img, flowDirection):

	dist_shock_CG = None
	dist_shield_CG = None

	#SHIELD

	if np.any(img_mask == [0, 255, 0]):
		shield_color1 = np.asarray([0, 230, 0])
		shield_color2 = np.asarray([1, 255, 1])
		mask_shield = cv2.inRange(img_mask, shield_color1, shield_color2)

		coordinates =cv2.findNonZero(mask_shield)
		x = [p[0][0] for p in coordinates]
		y = [p[0][1] for p in coordinates]
		centroid = (int(sum(x) / len(coordinates)), int(sum(y) / len(coordinates)))

		print(centroid)

		contours, hierarchy= cv2.findContours(mask_shield, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		#cnts = imutils.grab_contours(contours)
		c = max(contours, key=cv2.contourArea)
		c = np.vstack(c).squeeze()
		ext_values = []
		#c = np.array(cnts)


	# ext_values = []
	#
	# for i, idx in enumerate(c[:,0,1]):
	# 	if (i == centroid[1]):
	# 		ext_values.append(i)
	#
	# if flowDirection == "left":
	# 	ext_shock = [min(ext_values), centroid[1]]
	# if flowDirection == "right":
	# 	ext_shock = [max(ext_values), centroid[1]]


		xi = c[:,0]
		yi = c[:,1]
		x = xi
		y = yi-centroid[1]#[ind]

	    # adjust starting location
		if flowDirection == 'left':
			imin = y.argmin()
			y = np.roll(y,-imin)
			x = np.roll(x,-imin)
	        # remove non-function corner points
			dyGreaterThanZero = np.append(np.zeros(1),np.diff(y) > 1)
			okay = np.nonzero(dyGreaterThanZero)[0]
		else:
	        # remove non-function corner points
			dyLessThanZero = np.append(np.zeros(1),np.diff(y) < 0)
			okay = np.nonzero(dyLessThanZero)[0]


		print(y[okay])
	    # interpolate desired radial positions

		f = interp1d(y[okay], x[okay], kind='cubic')

		ext_shield = [int(f(0)), centroid[1]]
		print(ext_shield)

		# if flowDirection == "left":
		# 	ext_shield = [min(ext_values), centroid[1]]
		# 	print(ext_values)
		# if flowDirection == "right":
		# 	ext_shield = tuple(max(ext_values) [centroid[1]])



		#cv2.circle(img, ext_shield, 8, (255, 255, 0), -1)

		dist_shield_CG = abs(centroid[0]-ext_shield[0])


		contours, hierarchy = cv2.findContours(mask_shield,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		#cv2.drawContours(img,contours,-1,(255, 0, 0),5)
		cv2.circle(img, centroid, 5, (255, 0, 0), 2)


	# SHOCK
	if np.any(img_mask == [255, 0, 0]):
		shock_color1 = np.asarray([230 ,0, 0])
		shock_color2 = np.asarray([255, 0, 1])

		mask_shock = cv2.inRange(img_mask, shock_color1, shock_color2)

		cnts = cv2.findContours(mask_shock, cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)


		contours, hierarchy= cv2.findContours(mask_shock, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		#cnts = imutils.grab_contours(contours)
		c = max(contours, key=cv2.contourArea)
		c = np.vstack(c).squeeze()
		ext_values = []
		#c = np.array(cnts)


		for idx, i in enumerate(c[:,1]):
			if (i == centroid[1]):

				ext_values.append(c[idx,0])

		xi = c[:,0]
		yi = c[:,1]
		x = xi
		y = yi-centroid[1]#[ind]

	    # adjust starting location
		if flowDirection == 'left':
			imin = y.argmin()
			y = np.roll(y,-imin)
			x = np.roll(x,-imin)
	        # remove non-function corner points
			dyGreaterThanZero = np.diff(y) >2
			okay = np.nonzero(dyGreaterThanZero)[0]
		else:
	        # remove non-function corner points
			dyLessThanZero = np.append(np.zeros(1),np.diff(y) < 0)
			okay = np.nonzero(dyLessThanZero)[0]

		#x
	    # interpolate desired radial positions
		fShock = interp1d(y[okay], x[okay], kind='cubic')

		ext_shock = [int(fShock(0)), centroid[1]]
		print(ext_shock)
		#
		# ext_values = []
		#
		# for i, idx in enumerate(c[:,0,1]):
		# 	if (i == centroid[1]):
		# 		ext_values.append(i)
		#
		# if flowDirection == "left":
		# 	ext_shock = [min(ext_values), centroid[1]]
		# if flowDirection == "right":
		# 	ext_shock = [max(ext_values), centroid[1]]

		#cv2.circle(img, ext_shock, 8, (255, 255, 0), -1)

		dist_shock_CG = abs(centroid[0]-ext_shock[0])


#extLeft_CG = tuple(c[extLocLeft][0])

#extRight_CG = tuple(c[c[:, :, 0].argmax()][0])


	#cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
	#cv2.circle(img, extLeft, 8, (255, 0, 0), -1)


	#cv2.imshow("Robust", img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows("Robust")

	return dist_shock_CG, dist_shield_CG




if __name__ == "__main__":
	img_mask = cv2.imread('shock_detection/tests/frame_0012_out.png')
	img = cv2.imread('shock_detection/tests/frame_0012.png')
	flowDirection = "left"
	shock_extremity(img_mask, img, flowDirection)
