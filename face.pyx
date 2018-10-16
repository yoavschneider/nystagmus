import numpy as np
import cv2
from nyst import iris
from nyst import gradient
from nyst import geo
import dlib

from imutils import face_utils

class Tracker:
	h_eye_size_factor = 1.2
	v_eye_size_factor = 0.9
	eye_area_resolution = 200

	def __init__(self, track_left_iris=True,track_right_iris=True):
		self.track_left_iris = track_left_iris
		self.track_right_iris = track_right_iris
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor('resources/shape_predictor_68_face_landmarks.dat')
		self.has_face = False
		self.has_features = False
		self.has_irises = False

	def get_left_iris(self):
		x,y = self.left_iris
		return (int(x), int(y))
	def get_left_phase(self):
		return self.left_phase
	def get_right_iris(self):
		x,y = self.right_iris
		return (int(x), int(y))
	def get_left_eye_region(self):
		return self.left_eye_region
	def get_right_eye_region(self):
		return self.right_eye_region
	def get_left_eye_heatmap(self):
		return self.left_eye_heatmap
	def get_right_eye_heatmap(self):
		return self.right_eye_heatmap
	def get_right_phase(self):
		return self.right_phase
	def get_right_eye_borders(self):
		return ((self.rrx, self.rry), (self.rlx,self.rly))
	def get_left_eye_borders(self):
		return ((self.lrx, self.lry), (self.llx,self.lly))
	def get_nose_bridge(self):
		return ((self.c1x,self.c1y),(self.c2x,self.c2y),(self.c3x,self.c3y),(self.c4x,self.c4y))
	def get_vertical_phase(self):
		return self.vertical_phase
	def get_all_features(self):
		return self.all_features
	def get_face_horizontal_phase(self):
		return self.face_horizontal_phase
	def get_face_vertical_phase(self):
		return self.face_vertical_phase

	def read(self, img):
		self.img = img
		self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		if not self.has_face:
			self.__find_face()
		else:
			ok, face = self.tracker.update(self.gray)
			if ok:
				fx,fy,fw,fh = face
				self.fx,self.fy,self.fw,self.fh = int(fx),int(fy),int(fw),int(fh)
			else:
				print ("Lost face")
				self.has_face = False
				self.has_features = False
				self.has_irises = False
			if self.fw == 0 or self.fh == 0:
				print ("Lost face")
				self.has_face = False
				self.has_features = False
				self.has_irises = False

		if self.has_face:
			self.__locate_face_features()

			if self.track_right_iris:
				self.__locate_right_iris()
			if self.track_left_iris:
				self.__locate_left_iris()
			if not self.has_irises:
				self.has_irises = True

			self.__calculate_phases()

	def __locate_left_iris(self):
		w,h,y,x = self.__get_left_eye_area()
		self.left_eye_region = self.equ[y:y+h,x:x+w]

		if self.left_eye_region.size == 0:
			print ("Lost face")
			self.left_eye_heatmap = self.left_eye_region
			self.has_face = False
			return 

		result = iris.find_center(self.left_eye_region, self.fw, Tracker.eye_area_resolution)
		self.left_iris = (x + result[0], y + result[1])
		self.left_eye_heatmap = iris.get_heatmap()

	def __locate_right_iris(self):
		w,h,y,x = self.__get_right_eye_area()

		self.right_eye_region = self.equ[y:y+h,x:x+w]

		if self.right_eye_region.size == 0:
			print ("Lost face")
			self.right_eye_heatmap = self.right_eye_region
			self.has_face = False
			return 

		result = iris.find_center(self.right_eye_region, self.fw, Tracker.eye_area_resolution)

		self.right_iris = (x + result[0], y + result[1])

		self.right_eye_heatmap = iris.get_heatmap()

	def __calculate_phases(self):
		cdef double left_x, left_y, nose_centroid_x, nose_centroid_y, v_x, v_y, l_x, l_y, r_x, r_y
		left_x, left_y = self.left_iris
		right_x, right_y = self.right_iris
		nose_centroid_x, nose_centroid_y = geo.get_4_point_centroid(self.c1x,self.c1y,self.c2x,self.c2y,self.c3x,self.c3y,self.c4x,self.c4y)
		v_x, v_y, l_x, l_y, r_x, r_y = geo.get_phases_to_center_point(nose_centroid_x, nose_centroid_y,left_x, left_y,right_x, right_y)
		self.vertical_phase, self.left_phase, self.right_phase = (v_x, v_y), (l_x, l_y), (r_x, r_y)

		cdef double left_centroid_x, left_centroid_y, right_centroid_x, right_centroid_y
		cdef int p1,p2,p3,p4,p5,p6,p7,p8
		(p1,p2),(p3,p4),(p5,p6),(p7,p8) = self.all_features[0], self.all_features[1], self.all_features[2], self.all_features[3]
		left_centroid_x, left_centroid_y = geo.get_4_point_centroid(p1,p2,p3,p4,p5,p6,p7,p8)
		(p1,p2),(p3,p4),(p5,p6),(p7,p8) = self.all_features[13], self.all_features[14], self.all_features[15], self.all_features[16]
		right_centroid_x, right_centroid_y = geo.get_4_point_centroid(p1,p2,p3,p4,p5,p6,p7,p8)
		fv_x, fv_y, fh_x, fh_y = geo.get_phases_to_side_points(nose_centroid_x, nose_centroid_y,left_centroid_x, left_centroid_y,right_centroid_x, right_centroid_y)

		self.face_vertical_phase = fv_x, fv_y
		self.face_horizontal_phase = fh_x, fh_y

	def __locate_face_features(self):
		# Produce bluured and equalized images
		self.blurred = cv2.GaussianBlur(self.gray,(5,5), 0.005 * self.fw)

		# Use dlib to find eye edges and nose
		face_rect = dlib.rectangle(self.fx,self.fy,self.fx+self.fw,self.fy+self.fh)
		shape = self.predictor(self.blurred, face_rect)
		shape = face_utils.shape_to_np(shape)

		self.equ = self.blurred

		self.__update_features(shape)

		# equalize historgram for the pupil detection
		w,h,y,x = self.__get_left_eye_area()
		equ_left_eye_area = cv2.equalizeHist(self.blurred[y:y+h, x:x+w])
		self.equ[y:y+h, x:x+w] = equ_left_eye_area

		w,h,y,x = self.__get_right_eye_area()
		equ_right_eye_area = cv2.equalizeHist(self.blurred[y:y+h, x:x+w])
		self.equ[y:y+h, x:x+w] = equ_right_eye_area

		self.equ = (self.equ * 0.5).astype(np.uint8)

	def __find_face(self):
		faces = self.detector(self.gray, 0)

		if len(faces) >= 1:
			print("Found a face")
			
			face = Tracker.__get_largest_face(faces)
			self.fx,self.fy,self.fw,self.fh = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
			self.tracker = cv2.TrackerMedianFlow_create()
			ok = self.tracker.init(self.gray, (self.fx,self.fy,self.fw,self.fh))
			if not ok:
				self.has_face = False
				return

			self.has_face = True

	@staticmethod
	def __get_largest_face(faces):
		largest_face = faces[0]
		max_width = largest_face.right() - largest_face.left()

		for face in faces:
			if face.left() - face.right() > max_width:
				max_width = face.right() > max_width
				largest_face = face

		return largest_face


	def __get_left_eye_area(self):
		rw = (self.llx - self.lrx)
		rh = rw

		w = int (rw * Tracker.h_eye_size_factor)
		h = int (rh * Tracker.v_eye_size_factor)
		y = int ((self.lly + self.lry) / 2 - (h / 2))
		x = int ((self.llx + self.lrx) / 2 - (w / 2))

		return (w,h,y,x)

	def __get_right_eye_area(self):
		rw = self.rlx - self.rrx
		rh = rw

		w = int (rw * Tracker.h_eye_size_factor)
		h = int (rh * Tracker.v_eye_size_factor)
		y = int ((self.rly + self.rry) / 2 - (h / 2))
		x = int ((self.rlx + self.rrx) / 2 - (w / 2))

		return (w,h,y,x)

	def __update_features(self, shape):
		self.all_features = shape

		if self.has_features:
			for i in range(len(shape)):
				self.all_features[i] = (self.all_features[i] + shape[i]) / 2
		else:
			self.all_features = shape
			self.has_features = True

		self.lrx, self.lry = self.all_features[42] # Left eye right edge
		self.llx, self.lly = self.all_features[45] # Left eye left edge
		self.rrx, self.rry = self.all_features[36] # Right eye right edge
		self.rlx, self.rly = self.all_features[39] # Right eye left edge

		self.c1x, self.c1y = self.all_features[27] # Nose
		self.c2x, self.c2y = self.all_features[28] # Nose
		self.c3x, self.c3y = self.all_features[29] # Nose
		self.c4x, self.c4y = self.all_features[30] # Nose