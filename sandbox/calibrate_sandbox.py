#!/usr/bin/env python

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]

default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# built-in modules
import os
import sys
import getopt
from glob import glob

# local functions
def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def processImage(fn,debug_dir='./output/'):
    print('processing %s... ' % fn)
    img = cv.imread(fn, 0)
    if img is None:
        print("Failed to load", fn)
        return None

    h, w = img.shape[:2]

    assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
    found, corners = cv.findChessboardCorners(img, pattern_size)
    if found:
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

    if debug_dir:
        vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(vis, pattern_size, corners, found)
        _path, name, _ext = splitfn(fn)
        outfile = os.path.join(debug_dir, name + '_chess.png')
        cv.imwrite(outfile, vis)

    if not found:
        print('chessboard not found')
        return None

    print('           %s... OK' % fn)
    return (corners.reshape(-1, 2), pattern_points)

def getPose(fname,pattern,objp,mtx,dist):
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern,None)

    if ret == True:
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        corners2 = cv.cornerSubPix(gray,corners,(5,5),(-1,-1),term)

        # Find the rotation and translation vectors.
        success,direc,trans,ind = cv.solvePnPRansac(objp, corners2, mtx, dist)
        return (success,direc,trans)
    else:
        print("Could not find pattern corners :'(")
        return (False, False)
        



img_mask = './chessboard/left??.jpg'  # default
img_names = glob(img_mask)
debug_dir = './output/'
if debug_dir and not os.path.isdir(debug_dir):
    os.mkdir(debug_dir)
square_size = 1.0
threads_num = 8
pattern_size = (9, 6)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []
h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results

if threads_num <= 1:
    chessboards = [processImage(fn) for fn in img_names]
else:
    print("Run with %d threads..." % threads_num)
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool(threads_num)
    chessboards = pool.map(processImage, img_names)

chessboards = [x for x in chessboards if x is not None]
for (corners, pattern_points) in chessboards:
    img_points.append(corners)
    obj_points.append(pattern_points)

# calculate camera distortion
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

print("\nRMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

# undistort the image with the calibration
print('')
for fn in img_names if debug_dir else []:
    _path, name, _ext = splitfn(fn)
    img_found = os.path.join(debug_dir, name + '_chess.png')
    outfile = os.path.join(debug_dir, name + '_undistorted.png')

    img = cv.imread(img_found)
    if img is None:
        continue

    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

    # crop and save the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    print('Undistorted image written to: %s' % outfile)
    cv.imwrite(outfile, dst)

print('Done')


##mtx = np.array([[532.79536562,   0.,         342.45825163],
##             [  0.,         532.91928338, 233.90060514],
##             [  0.,           0.,           1.        ]])
##dist = [-2.81086258e-01,  2.72581009e-02,  1.21665908e-03, -1.34204274e-04,  1.58514023e-01]


pattern_size = (9, 6)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= 1.0

success, rvec, tvec= getPose(img_names[0],pattern_size,pattern_points,camera_matrix,dist_coefs)

# project 3D points to image plane
axis = np.float32([[3,0,-3],[0,0,0]])
img = cv.imread(img_names[0])
imgpts, jac = cv.projectPoints(axis, rvec, tvec, camera_matrix, dist_coefs)

img = cv.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[0].ravel()), (255,0,0), 5)
cv.imshow('img',img)
k = cv.waitKey(0) & 0xff

cv.destroyAllWindows()
#434.23962, 176.66481
