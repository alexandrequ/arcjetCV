import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep, interp1d

img = cv2.imread('shock_detection/tests/frame_0012_out.png')   # you can read in images with opencv


def extremity(img_mask, img, flowDirection):

	dist_shield_CG = None
	dist_shock_CG = None
	yShield_perc = None
	xShield_perc = None
	xShock_perc = None
	dist_shield_norm = None
	dist_shock_norm = None
	dist_shield_perc = None
	dist_shock_perc = None
	hasShield = 0
	hasShock = 0

	rnorms = [-.75,-.5,0,.5,0.75]

	#SHIELD

	sumMask = np.sum(img_mask)
	shield_color1 = np.asarray([0, 230, 0])
	shield_color2 = np.asarray([1, 255, 1])
	mask_shield = cv2.inRange(img_mask, shield_color1, shield_color2)

	hasShield = np.sum(mask_shield)

	if hasShield > 0.002*sumMask:

		#Centroid
		coordinates =cv2.findNonZero(mask_shield)
		x = [p[0][0] for p in coordinates]
		y = [p[0][1] for p in coordinates]
		centroid = (int(sum(x) / len(coordinates)), int(sum(y) / len(coordinates)))
		print(centroid)

		contours, hierarchy= cv2.findContours(mask_shield, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		cShield = max(contours, key=cv2.contourArea)
		(xb,yb,wb,dy) = cv2.boundingRect(cShield)
		c = np.vstack(cShield).squeeze()
		ext_values = []




		xi = c[:,0]
		yi = c[:,1]
		x = xi
		y = yi-centroid[1]#[ind]

		xFront = []
		yFront = []

	    # adjust starting location
		if flowDirection == 'left':
			# imin = y.argmin()
			# y = np.roll(y,-imin)
			# x = np.roll(xFront,-imin)
	        # remove non-function corner points
			dyGreaterThanZero = np.append(np.zeros(1),np.diff(y) > 0)
			okay = np.nonzero(dyGreaterThanZero)[0]
		else:
	        # remove non-function corner points
			dyLessThanZero = np.append(np.zeros(1),np.diff(y) < 0)
			okay = np.nonzero(dyLessThanZero)[0]

		x = x[okay]
		y = y[okay]


		iBefore = 0
		for idx, i in enumerate(y):

			if (y ==i).sum() > 1:
				jdx = np.where(y == i)
				if flowDirection == 'left':
					goodIdx = np.where(jdx == min(x[jdx]))
				if flowDirection == 'right':
					goodIdx = np.where(jdx == max(x[jdx]))

				jdx = np.delete(jdx, goodIdx)
				x = np.delete(x,jdx)
				y = np.delete(y,jdx)
			# if abs(i-iBefore)  < 1:

			# 	y = np.delete(y,idx)
			# iBefore = y[idx-1]

	    # interpolate desired radial positions

		fShield = interp1d(y, x, kind='cubic')
		ext_shield = [int(fShield(0)), centroid[1]]

		R_px = dy/2.

		try:
			xShield_perc = fShield(np.array(rnorms)*R_px)
			dist_shield_perc = abs(centroid[0]*np.ones(5)-xShield_perc)
		except:
		 	dist_shield_perc = np.empty(5)


		yShield_perc = np.array(rnorms)*R_px

		#cv2.circle(img, ext_shield, 8, (255, 255, 0), -1)

		dist_shield_CG = abs(centroid[0]-ext_shield[0])
		dist_shield_norm = dist_shield_CG/dy
		#cv2.drawContours(img_mask, cShield, -1, (0,255,255), 3)

		contours, hierarchy = cv2.findContours(mask_shield,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		#cv2.drawContours(img,contours,-1,(255, 0, 0),5)
		cv2.circle(img, centroid, 5, (255, 0, 0), 2)


	# SHOCK

	shock_color1 = np.asarray([230 ,0, 0])
	shock_color2 = np.asarray([255, 0, 1])
	mask_shock = cv2.inRange(img_mask, shock_color1, shock_color2)
	hasShock = np.sum(mask_shock)

	if hasShock > 0.002*sumMask:


		contours, hierarchy= cv2.findContours(mask_shock, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		#cnts = imutils.grab_contours(contours)
		cShock = max(contours, key=cv2.contourArea)
		c = np.vstack(cShock).squeeze()
		ext_values = []
		#c = np.array(cnts)


		# for idx, i in enumerate(c[:,1]):
		# 	if (i == centroid[1]):
		#
		# 		ext_values.append(c[idx,0])

		xi = c[:,0]
		yi = c[:,1]
		x = xi
		y = yi-centroid[1]#[ind]


		xFront = []
		yFront = []

	    # adjust starting location
		if flowDirection == 'left':
			# imin = y.argmin()
			# y = np.roll(y,-imin)
			# x = np.roll(xFront,-imin)
	        # remove non-function corner points
			dyGreaterThanZero = np.append(np.zeros(1),np.diff(y) > 0)
			okay = np.nonzero(dyGreaterThanZero)[0]
		else:
	        # remove non-function corner points
			dyLessThanZero = np.append(np.zeros(1),np.diff(y) < 0)
			okay = np.nonzero(dyLessThanZero)[0]

		x = x[okay]
		y = y[okay]

		iBefore = 0
		for idx, i in enumerate(y):

			if (y ==i).sum() > 1:
				jdx = np.where(y == i)
				if flowDirection == 'left':
					goodIdx = np.where(jdx == min(x[jdx]))
				if flowDirection == 'right':
					goodIdx = np.where(jdx == max(x[jdx]))

				jdx = np.delete(jdx, goodIdx)
				x = np.delete(x,jdx)
				y = np.delete(y,jdx)

		fShock = interp1d(y, x, kind='cubic')
		ext_shock = [int(fShock(0)), centroid[1]]

		try:
			xShock_perc = fShock(np.array(rnorms)*R_px)
			dist_shock_perc = abs(centroid[0]*np.ones(5)-xShock_perc)
		except:
			dist_shock_perc = np.empty(5)


		yShield = np.arange(min(yShield_perc), max(yShield_perc), 0.01)
		try:
			xShield = [centroid[0]*np.ones(len(yShield)) - fShield(yShield)]
		except:
			xShield = np.empty(len(yShield))
		try:
			xShock = [centroid[0]*np.ones(len(yShield)) - fShock(yShield)]
		except:
			xShock = np.empty(len(yShield))




		dist_shock_CG = abs(centroid[0]-ext_shock[0])
		dist_shock_norm = dist_shock_CG/dy




	return dist_shock_norm, dist_shield_norm, yShield_perc, dist_shock_perc, dist_shield_perc, img_mask




if __name__ == "__main__":
	img_mask = cv2.imread('shock_detection/tests/frame_0012_out.png')
	img = cv2.imread('shock_detection/tests/frame_0012.png')
	flowDirection = "left"
	extremity(img_mask, img, flowDirection)
