import numpy as np
import math
cimport cython
cimport numpy as np
from libc.math cimport sqrt
from libc.math cimport atan
from libc.math cimport cos

@cython.cdivision(True)
def get_4_point_centroid(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4):
	cdef double c_x = (x1 + x2 + x3 + x4) / 4
	cdef double c_y = (y1 + y2 + y3 + y4) / 4

	return c_x, c_y

# c_x, c_y - nose centroid
# left_x/y, right_x/y - side centroids
@cython.cdivision(True)
def get_phases_to_side_points(double c_x, double c_y, double left_x, double left_y, double right_x, double right_y):
	# Line between side points
	cdef double a1 = (left_y - right_y) / (left_x - right_x)
	cdef double b1 = right_y - a1 * right_x
	
	cdef double a2, b2, px, py

	# Calculate line between irises and perpendicular line from nose centroid
	if a1 != 0:
		a2 = 1 / a1
		b2 = c_y - a2 * c_x

		px = (b2 - b1) / (a1 - a2)
		py = (px * a1 + b1)
	else:
		px = c_x
		py = left_y

	# calculate phases
	cdef double v_phase_x, v_phase_y, h_phase_x, h_phase_y

	# vertical phase
	v_phase_y = c_y - py
	v_phase_x = c_x - px

	# horizontal phase
	h_phase_y = py - ((right_y + left_y) / 2)
	h_phase_x = px - ((right_x + left_x) / 2)

	return v_phase_x, v_phase_y, h_phase_x, h_phase_y

# c_x/y - nose centroid
@cython.cdivision(True)
def get_phases_to_center_point(double c_x, double c_y, double left_x, double left_y, double right_x, double right_y):
	# Line between side points
	cdef double a1 = (left_y - right_y) / (left_x - right_x)
	cdef double b1 = right_y - a1 * right_x
	
	cdef double a2, b2, px, py

	# Calculate line between irises and perpendicular line from nose centroid
	if a1 != 0:
		a2 = 1 / a1
		b2 = c_y - a2 * c_x

		px = (b2 - b1) / (a1 - a2)
		py = (px * a1 + b1)
	else:
		px = c_x
		py = left_y

	# calculate phases
	cdef double v_phase_x, v_phase_y, l_phase_x, l_phase_y, r_phase_x, r_phase_y

	# vertical phase
	v_phase_y = c_y - py
	v_phase_x = c_x - px

	# left phase
	l_phase_y = left_y - py
	l_phase_x = left_x - px

	# right phase
	r_phase_y = right_y - py
	r_phase_x = right_x - px

	return v_phase_x, v_phase_y, l_phase_x, l_phase_y, r_phase_x, r_phase_y

# normalize phase by distance (same lengths in images in different distances should be the same)
@cython.cdivision(True)
def normalize_phase(double dx, double dy, double Z, double F_x, double F_y):
	dx = dx * Z / F_x
	dy = dy * Z / F_y

	return dx, dy

@cython.cdivision(True)
def get_depth_from_similar_triangle(double dx, double dy, double D, double F_x):
	cdef double alpha, DX

	alpha = atan(dy/dx)
	DX = cos(alpha) * D

	return  F_x * DX/dx