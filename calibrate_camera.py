import argparse
import numpy as np
import cv2
import time
import nyst.stream as stream

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--camera", help="camera port number")
parser.add_argument("-n", "--number", help="number of calibration images to be taken", type=int, default=15)
parser.add_argument("-om", "--matrix", help="save matrix as file, default 'camera-matrix.txt'", default='data/camera-matrix.txt')
parser.add_argument("-od", "--distortion", help="save distortion factors as file, default 'distortion-factors.txt'", default="data/distortion-factors.txt")

def current_milli_time():
    return int(round(time.time() * 1000))

args, leftovers = parser.parse_known_args()

cap = stream.WebcamVideoStream(src=args.camera, width=800, height=600).start()

run = True

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

count = 0

last_image_taken = current_milli_time()

while run:
	img, _ = cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

	if ret == True and current_milli_time() - last_image_taken > 1000:
		last_image_taken = current_milli_time()
		objpoints.append(objp)

		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners2)

		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
		count = count + 1

	cv2.imshow('img',img)

	k = cv2.waitKey(1) & 0xff
	if k == 27:
		cap.stop()
		break
	if count >= args.number:
		break

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print(ret)

h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))

while run:
	img, _ = cap.read()
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
	cv2.imshow('img',dst)
	k = cv2.waitKey(1) & 0xff
	if k == 27:
		cap.stop()
		break

print ("Written to " , args.distortion)
np.savetxt(args.distortion, dist)
print ("Written to " , args.matrix)
np.savetxt(args.matrix, newcameramtx)