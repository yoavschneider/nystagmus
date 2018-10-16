from __future__ import print_function

import cv2
import sys
import time

from matplotlib import pyplot as plt
import numpy as np

import nyst.stream as stream
import nyst.projection as projection
import nyst.face as face

frame_rate = 29
frame_duration = 1000 / frame_rate

def current_time_sec():
	return int(round(time.time()))

def write(f, line):
	print(line, file=f)

def stop():
	cap.stop()

def start(filename, factors_file, camera, seconds, sensor_width_mm, sensor_height_mm):
	global cap

	# Setup capture
	cap = stream.WebcamVideoStream(src=camera, width=800, height=600).start()
	img, _ = cap.read()
	height, width, _ = img.shape
	print ("Video Resolution: " + str(width) + ", " + str(height))	

	# Load 3D model
	model = projection.Projection(width=width, sensor_width_mm=sensor_width_mm, height=height, sensor_height_mm=sensor_height_mm)

	print("Sensor Size: " + str(sensor_width_mm) + ", " + str(sensor_height_mm))

	# Set Tracker
	tracker = face.Tracker(True, True)

	cv2.startWindowThread()
	cv2.namedWindow("image")

	fps_start_time = time.time()
	frame_count = 0

	cdef int run 
	global run
	run = 1

	K = model.get_camera_matrix()
	dist_factors = model.get_dist_factors()

	start_time = current_time_sec()
	with open(filename, "w+") as f:
		while run > 0:
			if current_time_sec() - start_time > seconds:
				cap.stop()
				break

			t = time.time()
			frame, fresh = cap.read()
			timestamp = time.time()

			img = cv2.undistort(frame.copy(), K, dist_factors, None, K)

			tracker.read(img)
			if tracker.has_face:
				left_eye_region = tracker.get_left_eye_region()
				left_iris = tracker.get_left_iris()
				right_eye_region = tracker.get_right_eye_region()
				right_iris = tracker.get_right_iris()

				lx, ly = left_iris
				rx, ry = right_iris
				z = model.get_plane_depth_from_points(lx, ly, rx, ry, 63)

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

				if fresh:
					line = "{};{};{};{};{};{};{};{}".format(timestamp, f_h_phase_x, l_phase_x, r_phase_x, f_v_phase_y, v_phase_y, (x1,y1,z), (x2,y2,z))
					write(f, line)

			# print frame rate
			if (frame_count % 30 == 0):
				print(frame_count / (time.time() - fps_start_time))
				fps_start_time = time.time()
				frame_count = 0

			frame_count = frame_count + 1

			k = cv2.waitKey(1) & 0xff
			if k == 27:
				cap.stop()
				break

	cv2.destroyAllWindows()

