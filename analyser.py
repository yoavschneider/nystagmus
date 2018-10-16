import argparse
import numpy as np
import pandas as pd
import time, math

from matplotlib import rcParams

labelsize = 16
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 

import rigid_transform as transform

import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser()
parser.add_argument("-nyst", "--nyst")
parser.add_argument("-tobii", "--tobii")
parser.add_argument("-factors", "--factors")
parser.add_argument("-t", "--title", type=int)
parser.add_argument("-f", "--filename")

# Dictionary for a shorter "title" parameter
title_dict = {
	0: "Left-Fast",
	1: "Left-Slow",
	2: "Down-Fast",
	3: "Down-Slow",
	4: "Reading an Article",
}

args, leftovers = parser.parse_known_args()

# Screen Size in the Experiment
SCREEN_WIDTH_MM = 517 
SCREEN_HEIGHT_MM = 324

# Nyst values
nyst_timestamps = []
nyst_left_phase_x = []
nyst_right_phase_x = []
nyst_both_phase_y = []
nyst_face_phase_x = []
nyst_face_phase_y = []
nyst_left_eye_pos = []
nyst_right_eye_pos = []

# from calibration
nyst_normalized_left_x = []
nyst_normalized_right_x = []
nyst_normalized_both_y = []

# newly calculated
nyst_new_normalized_left_x = []
nyst_new_normalized_right_x = []
nyst_new_normalized_both_y = []

left_eye_factors = []
right_eye_factors = []
vertical_factors = []

# Tobii values
tobii_timestamps = []
# Screen
tobii_left_gaze_x = []
tobii_left_gaze_y = []
tobii_right_gaze_x = []
tobii_right_gaze_y = []
# averaged vertical gaze
tobii_both_gaze_y = []

# Gaze points in user Coordinate System
tobii_left_gaze_pos = []
tobii_right_gaze_pos = []

# Eyes in user coordinate system
tobii_left_eye_pos = []
tobii_right_eye_pos = []

left_eye_distances = []
right_eye_distances = []

resampled_tobii_left_eye_x = []
resampled_tobii_right_eye_x  = []
resampled_tobii_left_eye_x  = []
resampled_tobii_right_eye_x  = []
resampled_tobii_both_eyes_y = []
resampled_nyst_left_eye_x  = []
resampled_nyst_right_eye_x  = []
resampled_nyst_both_eyes_y  = []

def parse_tuple_float(string):
	return [float (val) for val in string.replace("(","").replace(")","").split(", ")]

def parse_nyst_line(line):
	values = line.split(";")
	timestamp = float(values[0])
	face_phase_x = float(values[1])
	left_phase_x = float(values[2])
	right_phase_x = float(values[3])
	face_phase_y = float(values[4])
	both_phase_y = float(values[5])

	left_iris_pos = parse_tuple_float(values[6])
	right_iris_pos = parse_tuple_float(values[7])

	nyst_timestamps.append(timestamp)
	
	nyst_face_phase_x.append(face_phase_x)
	nyst_left_phase_x.append(left_phase_x)
	nyst_right_phase_x.append(right_phase_x)

	nyst_face_phase_y.append(face_phase_y)
	nyst_both_phase_y.append(both_phase_y)

	nyst_left_eye_pos.append(left_iris_pos)
	nyst_right_eye_pos.append(right_iris_pos)

def parse_tobii_line(line):
	values = line.split(";")

	# DO WHEN RECORDING!!!
	if values[1] == "(nan, nan)":
		return

	timestamp = float(values[0])
	# gaze points in screen coordinates (0-1)
	left_screen_pos = parse_tuple_float(values[1])
	right_screen_pos = parse_tuple_float(values[2])
	# gaze points in user coordinates (3D)
	left_gaze_pos = parse_tuple_float(values[5])
	right_gaze_pos = parse_tuple_float(values[6])
	# eye positions in user coordinates (3D)
	left_eye_pos = parse_tuple_float(values[3])
	right_eye_pos = parse_tuple_float(values[4])

	tobii_timestamps.append(timestamp)
	tobii_left_gaze_x.append(left_screen_pos[0])
	tobii_left_gaze_y.append(left_screen_pos[1])
	tobii_right_gaze_x.append(right_screen_pos[0])
	tobii_right_gaze_y.append(right_screen_pos[1])
	tobii_both_gaze_y.append((left_screen_pos[1] + right_screen_pos[1]) / 2)
	tobii_left_gaze_pos.append(left_gaze_pos)
	tobii_right_gaze_pos.append(right_gaze_pos)
	tobii_left_eye_pos.append(left_eye_pos)
	tobii_right_eye_pos.append(right_eye_pos)

def load_nyst_calibration_factors(file):
	global left_eye_factors
	global right_eye_factors
	global vertical_factors
	left_eye_factors = [float(val) for val in file.readline().split(";")]
	right_eye_factors = [float(val) for val in file.readline().split(";")]
	vertical_factors = [float(val) for val in file.readline().split(";")]

def normalize_nyst_data():
	values = zip(nyst_left_phase_x, nyst_right_phase_x, nyst_both_phase_y, nyst_face_phase_x, nyst_face_phase_y)
	for left_eye, right_eye, both_eyes, horizontal_face, vertical_face in values:
		nyst_normalized_left_x.append(horizontal_face * left_eye_factors[0] + left_eye * left_eye_factors[1] + left_eye_factors[2])
		nyst_normalized_right_x.append(horizontal_face * right_eye_factors[0] + right_eye * right_eye_factors[1] + right_eye_factors[2])
		nyst_normalized_both_y.append(vertical_face * vertical_factors[0] + both_eyes * vertical_factors[1] + vertical_factors[2])

def load_tobii_screen_data(file):
	global tobii_screen_top_left
	global tobii_screen_top_right
	global tobii_screen_bottom_left
	global tobii_screen_bottom_right

	tobii_screen_top_left = parse_tuple_float(file.readline())
	tobii_screen_top_right = parse_tuple_float(file.readline())
	tobii_screen_bottom_left = parse_tuple_float(file.readline())
	tobii_screen_bottom_right = parse_tuple_float(file.readline())

def load_screen_data(file):
	global screen_width_mm
	global screen_height_mm

	values = file.readline().split(" ")
	screen_width_mm, screen_height_mm = float(values[0]), float(values[1])

def euclidean_distance(p1,p2):
	return math.sqrt(math.pow(p1[0]-p2[0],2) + math.pow(p1[1]-p2[1],2) + math.pow(p1[2]-p2[2],2))

# Calculate the transformation from Tobii's user coordinate system (origin in the eye tracker) to the screen
# coordinate system (origin in the top left corner)
def calculate_tobii_to_screen_transformation():
	global tobii_screen_rotation
	global tobii_screen_translation
	screen = np.array([[0,0,0],[screen_width_mm,0,0],[0,screen_height_mm,0]])
	tobii_screen = np.array([tobii_screen_top_left,tobii_screen_top_right,tobii_screen_bottom_left])

	tobii_screen_rotation, tobii_screen_translation = transform.rigid_transform_3D(tobii_screen, screen)
	

	[0,0,0],[screen_width_mm,0,0],[0,screen_height_mm,0],[screen_width_mm,screen_height_mm,0]

	error1 = euclidean_distance(np.dot(tobii_screen_rotation, tobii_screen_top_left) + tobii_screen_translation, [0,0,0])
	error2 = euclidean_distance(np.dot(tobii_screen_rotation, tobii_screen_top_right) + tobii_screen_translation, [screen_width_mm,0,0])
	error3 = euclidean_distance(np.dot(tobii_screen_rotation, tobii_screen_bottom_left) + tobii_screen_translation, [0,screen_height_mm,0])
	error4 = euclidean_distance(np.dot(tobii_screen_rotation, tobii_screen_bottom_right) + tobii_screen_translation, [screen_width_mm,screen_height_mm,0])
	
	#print("Mean Screen Corners Error = " + str((error1 + error2 + error3 +error4)/4))

# returns a copy containing only the provided indexes
# l - a list
# indexes -  a list of indexes to keep
def filter(l,indexes):
	new = []

	i = 0
	indexes = indexes.copy()
	indexes.reverse()
	keep = indexes.pop()
	for val in l:
		if i == keep:
			new.append(val)
			if indexes:
				keep = indexes.pop()
			else:
				break
		i = i + 1

	return new

# Input: x, y lists of timestamps, offset
# return a list of indexs in x and y such that for each remaining x there
# is a matching entry in y with a minimal time difference of less than OFFSET
def match(x, y, min_offset):
	if len(x) == 0 or len(y) == 0:
		print("One of the lists is empty")
		return [],[]

	x_indexes = []
	y_indexes = []

	x_end_index = len(x) - 1
	y_end_index = len(y) - 1

	x_index = 0
	y_index = 0
	
	last_x = -1
	last_y = -1

	while x_index <= x_end_index and y_index <= y_end_index:
		current_x = x[x_index]
		current_y = y[y_index]

		if current_x < current_y:
			# Look until current_x > current_y, end was reached or offset is small enough
			while current_x < current_y and abs(current_x - current_y) > min_offset and x_index <= x_end_index:
				x_index = x_index + 1
				if x_index <= x_end_index:
					current_x = x[x_index]

			# No match was found
			if current_x < current_y and abs(current_x - current_y) > min_offset:
				return x_indexes, y_indexes

			# Check the last x if possible
			if x_index > 0 and x_index - 1 > last_x:
				offset_previous = abs(x[x_index - 1] - current_y)
				offset_current = abs(x[x_index] - current_y)

				if offset_previous < offset_current:
					x_index = x_index - 1

			x_indexes.append(x_index)
			y_indexes.append(y_index)
			last_x = x_index
			last_y = y_index
			x_index = x_index + 1
			y_index = y_index + 1
		else:
			while current_y < current_x and abs(current_y - current_x) > min_offset and y_index <= y_end_index:
				y_index = y_index + 1
				if y_index <= y_end_index:
					current_y = y[y_index]

			# No match was found
			if current_y < current_x and abs(current_x - current_y) > min_offset:
				return x_indexes, y_indexes

			# Check the last y if possible
			if y_index > 0 and y_index - 1 > last_y:
				offset_previous = abs(y[y_index - 1] - current_x)
				offset_current = abs(y[y_index] - current_x)

				if offset_previous < offset_current:
					y_index = y_index - 1

			x_indexes.append(x_index)
			y_indexes.append(y_index)
			last_x = x_index
			last_y = y_index
			x_index = x_index + 1
			y_index = y_index + 1


	return x_indexes, y_indexes

# filter and calculate calculate error using error_func 
def filter_and_get_error(x,y,filter_x,filter_y,error_func):
	error = []

	x,y = filter(x, filter_x), filter(y, filter_y)

	for i in range(len(x)):
		error.append(error_func(x[i], y[i]))

	return error
	

def simple_horizontal_error_in_mm(x,y):
	return abs(x - y) * SCREEN_WIDTH_MM

def simple_vertical_error_in_mm(x,y):
	return abs(x - y) * SCREEN_HEIGHT_MM


# Load from files
with open(args.nyst, "r") as nyst_data:
	for line in nyst_data:
		parse_nyst_line(line)

with open(args.tobii, "r") as tobii_data:
	for line in tobii_data:
		parse_tobii_line(line)

with open(args.factors, "r") as factors:
	load_nyst_calibration_factors(factors)

normalize_nyst_data()

with open("data/tobii_screen_data.txt") as tobii_screen:
	load_tobii_screen_data(tobii_screen)

with open("data/screen.txt") as screen:
	load_screen_data(screen)

calculate_tobii_to_screen_transformation()

tobii_indexes, nyst_indexes = match(tobii_timestamps,nyst_timestamps, 0.05)


# Translate to screen coordinate system and calculate euclidean distance
def dist_error(pos1, pos2):
	pos1 = np.dot(tobii_screen_rotation, pos1) + tobii_screen_translation

	return euclidean_distance(pos1,pos2)

horizontal_left_eye_error = filter_and_get_error(tobii_left_gaze_x, nyst_normalized_left_x, tobii_indexes, nyst_indexes, simple_horizontal_error_in_mm)
horizontal_right_eye_error = filter_and_get_error(tobii_right_gaze_x, nyst_normalized_right_x, tobii_indexes, nyst_indexes, simple_horizontal_error_in_mm)
vertical_both_eyes_error = filter_and_get_error(tobii_both_gaze_y, nyst_normalized_both_y, tobii_indexes, nyst_indexes, simple_vertical_error_in_mm)
left_eye_pos_error = filter_and_get_error(tobii_left_eye_pos, nyst_left_eye_pos, tobii_indexes, nyst_indexes, dist_error)
right_eye_pos_error = filter_and_get_error(tobii_right_eye_pos, nyst_right_eye_pos, tobii_indexes, nyst_indexes, dist_error)

data_1 = [horizontal_left_eye_error, horizontal_right_eye_error, vertical_both_eyes_error]
data_2 = [left_eye_pos_error, right_eye_pos_error]
labels_1 = ["Left Eye X", "Right Eye X", "Both Eyes Y"]
labels_2 = ["Left Position", "Right Position"]

## PLOT ##

fig, axes = plt.subplots(ncols=2, figsize=(20,2.2))
axes[0].set_ylabel('Error in millimeters')
axes[1].set_ylabel('Error in millimeters')
axes[0].set_title('N = ' + str(len(horizontal_left_eye_error)), fontsize=20)
axes[1].set_title('N = ' + str(len(left_eye_pos_error)), fontsize=20)
axes[0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
axes[1].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
bp1 = axes[0].boxplot(data_1, labels=labels_1, showmeans=True, meanline=True)
bp2 = axes[1].boxplot(data_2, labels=labels_2, showmeans=True, meanline=True)

plt.setp(bp1['fliers'], color='red', marker='+')
plt.setp(bp2['fliers'], color='red', marker='+')

manager = plt.get_current_fig_manager()
manager.resize(1400, 500)

plt.savefig(args.filename + '.png', bbox_inches='tight')

with open(args.filename + 'means.csv', "w+") as f:
	line = "{};{};{};{};{}".format('horizontal_left_eye_error','horizontal_right_eye_error','vertical_both_eyes_error','left_eye_pos_error','right_eye_pos_error')
	print(line, file=f)
	line = "{};{};{};{};{}".format(np.around(np.mean(horizontal_left_eye_error), decimals=2), np.around(np.mean(horizontal_right_eye_error), decimals=2),np.around(np.mean(vertical_both_eyes_error),decimals=2),np.around(np.mean(left_eye_pos_error),decimals=2), np.around(np.mean(right_eye_pos_error),decimals=2))
	print(line, file=f)
