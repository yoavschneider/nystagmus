import numpy as np
cimport numpy as np

cimport cython

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from nyst import geo

class Projection():
	def __init__(self, width, sensor_width_mm, height, sensor_height_mm, k_filename="data/camera-matrix.txt", d_filename="data/distortion-factors.txt", r_filename="data/rotation-matrix.txt", t_filename="data/translation-vector.txt", s_filename="data/screen.txt"):
		self.K = np.asmatrix(np.loadtxt(k_filename))
		self.R = np.asmatrix(np.loadtxt(r_filename))
		self.T = np.asmatrix(np.loadtxt(t_filename)).transpose()
		self.KI = self.K.I
		self.RI = self.R.I
		self.dist = np.loadtxt(d_filename)
		self.screen = np.loadtxt(s_filename)
		self.ref_points = np.loadtxt("data/ex_model.txt")
		self.error = np.loadtxt("data/error.txt")
		self.sensor_width_mm = sensor_width_mm
		self.sensor_height_mm = sensor_height_mm
		self.sensor_width_pixel = width
		self.sensor_height_pixel = height

		self.F_x_mm = self.K.item(0,0) * sensor_width_mm / width
		self.F_y_mm = self.K.item(1,1) * sensor_height_mm / height

	def get_camera_matrix(self):
		return self.K

	def get_dist_factors(self):
		return self.dist

	def get_normalized_phase(self, double x_phase, double y_phase, double Z):
		return geo.normalize_phase(x_phase, y_phase, Z, self.F_x_mm, self.F_y_mm)

	# Get depth of a plane given two points and relative Z
	def get_plane_depth_from_points(self, int x1, int y1, int x2, int y2, double D):
		cdef double f_x, alpha, dx, dy, DX, DY, ret
		
		dx = (x1 - x2) * (self.sensor_width_mm / self.sensor_width_pixel)
		dy = (y1 - y2) * (self.sensor_height_mm / self.sensor_height_pixel)

		return geo.get_depth_from_similar_triangle(dx, dy, D, self.F_x_mm)

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def get_world_position(self, int x, int y, double Z):
		cdef np.ndarray[np.double_t, ndim=2] point = np.zeros([3, 1], dtype=np.double)
		cdef double[:,:] T = self.T
		point[0][0] = x
		point[1][0] = y
		point[2][0] = 1

		np.dot(self.KI, point, out=point)
		np.multiply(point, Z, out=point)
		point[0][0] = point[0][0] - T[0][0]
		point[1][0] = point[1][0] - T[1][0]
		point[2][0] = point[2][0] - T[2][0]
		
		np.dot(self.RI, point, out=point)
		
		return point.item(0), point.item(1), point.item(2)

	def start_visualization(self):
		plt.ion()
		fig = plt.figure()
		self.ax = fig.add_subplot(111, projection='3d')
		self.ax.set_xlabel('X Label')
		self.ax.set_ylabel('Y Label')
		self.ax.set_zlabel('Z Label')
		self.ax.view_init(elev=-80, azim=-90)
		self.ax.set_zlim3d(-100, 500)

	def visualize_points(self, points):
		cdef double cx, cy, cz
		cdef double x, y, z

		# Screen
		plt.cla()
		self.plot_screen(self.ax, 40)

		for x,y,z in points:
			cx, cy, cz = self.get_world_position(x,y,z)
			self.ax.scatter(cx, cy, cz, c='g')
		plt.draw()
		plt.pause(0.01)


	def visualize(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		
		# Camera
		self.plot_camera(ax)

		# Screen
		self.plot_screen(ax)
		self.plot_reference_points(ax)

		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')

		ax.view_init(elev=-80, azim=-90)

		error = self.error.item(0)

		plt.title(r'Reprojection error: ' + str(error), fontsize=15)
		plt.show()

	# Plot a line from the camera center forward
	def plot_camera(self, ax):
		x, y = self.K.item(0,2), self.K.item(1,2)
		cx, cy, cz = self.get_world_position(x,y,1)
		ax.scatter(cx, cy, cz, c='g',)

		for i in range(2, 10):
			cx, cy, cz = self.get_world_position(x,y,i * 5)
			ax.scatter(cx, cy, cz, c='r')

	# plot the screen as blue points
	def plot_screen(self,ax, n=10):
		cdef int width
		cdef int height   
		cdef int x
		cdef int y

		lx = []
		ly = []
		lz = []
		width = int(self.screen.item(0))
		height = int(self.screen.item(1))

		for x in range(0, width + n,n):
			for y in range(0, height + n, n):
				lx.append(x)
				ly.append(y)
				lz.append(0)
		
		ax.scatter(lx,ly,lz,c='b',marker='^')

	# plot the detected points of the aruco markers
	def plot_reference_points(self, ax):
		for x,y,z in self.ref_points:
			ax.scatter(x,y,z,c='r')
