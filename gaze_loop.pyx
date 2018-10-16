import cv2
import sys
import time

from matplotlib import pyplot as plt
import numpy as np

import nyst.stream as stream
import nyst.projection as projection
import nyst.face as face

current_milli_time = lambda: int(round(time.time() * 1000))
current_sec_time = lambda: int(round(time.time()))

frame_rate = 29
frame_duration = 1000 / frame_rate
last_frame = current_milli_time()

def fps():
	global last_frame
	time_passed = current_milli_time() - last_frame
	last_frame = current_milli_time()

	if time_passed < frame_duration:
		time.sleep(0.001 * (frame_duration - time_passed))

def stop():
	global run
	global cap
	run = 0
	cap.stop()

def start(gaze_tracker, camera, sensor_width_mm, sensor_height_mm):
	global frame_rate
	global frame_duration
	global cap

	cap = stream.WebcamVideoStream(camera).start()

	time.sleep(2)

	img, _ = cap.read()
	height, width, _ = img.shape
	print ("Video Resolution: " + str(width) + ", " + str(height))	

	model = projection.Projection(width=width, sensor_width_mm=sensor_width_mm, height=height, sensor_height_mm=sensor_height_mm)

	print("Sensor Size: " + str(sensor_width_mm) + ", " + str(sensor_height_mm))

	tracker = face.Tracker(True, True)

	cv2.startWindowThread()
	cv2.namedWindow("image")

	start_time = time.time()
	frame_count = 0

	last_frame = current_milli_time()

	cdef int run 
	global run
	run = 1

	K = model.get_camera_matrix()
	dist_factors = model.get_dist_factors()

	has_offset = False
	started = False

	cdef double offset_x, offset_y

	# model.start_visualization()

	while run > 0:
		frame, fresh = cap.read()

		img = cv2.undistort(frame, K, dist_factors, None, K)
		tracker.read(img)

		if tracker.has_face:			
			left_eye_region = tracker.get_left_eye_region()
			left_iris = tracker.get_left_iris()
			right_eye_region = tracker.get_right_eye_region()
			right_iris = tracker.get_right_iris()

			lx, ly = left_iris
			rx, ry = right_iris

			cv2.circle(img, (lx,ly),2,(0,0,255), -1)
			cv2.circle(img, (rx,ry),2,(0,255,0), -1)
			z = model.get_plane_depth_from_points(lx, ly, rx, ry, 63)
			# model.visualize_points([(lx, ly, z),(rx, ry, z)])

			x1,y1,z = model.get_world_position(lx, ly, z)
			x2,y2,z = model.get_world_position(rx, ry, z)

			l_phase_x, l_phase_y = tracker.get_left_phase()
			r_phase_x, r_phase_y = tracker.get_right_phase()
			v_phase_x, v_phase_y = tracker.get_vertical_phase()

			fh_x, fh_y = tracker.get_face_horizontal_phase()
			fv_x, fv_y = tracker.get_face_vertical_phase()

			l_phase_x, _ = model.get_normalized_phase(l_phase_x, l_phase_y, z)
			r_phase_x, _ = model.get_normalized_phase(r_phase_x, r_phase_y, z)
			_, v_phase_y = model.get_normalized_phase(v_phase_x, v_phase_y, z)
			f_h_phase_x, _ = model.get_normalized_phase(fh_x, fh_y, z)
			_, f_v_phase_y = model.get_normalized_phase(fv_x, fv_y, z)

			# record calibration data
			if fresh:
				gaze_tracker.add_data(l_phase_x, r_phase_x, v_phase_y, f_h_phase_x, f_v_phase_y)

				# cv2.imshow("image", img)

		# print frame rate
		if (frame_count % 100 == 0):
			print(frame_count / (time.time() - start_time))
			start_time = time.time()
			frame_count = 0

		frame_count = frame_count + 1

		k = cv2.waitKey(1) & 0xff
		if k == 27:
			cap.stop()
			break

	cv2.destroyAllWindows()

