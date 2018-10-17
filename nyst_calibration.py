import sys, random, argparse, time, os

from PyQt5.QtWidgets import QWidget, QApplication, QDesktopWidget
from PyQt5.QtGui import QPainter, QColor, QImage, QPen
from PyQt5.QtCore import Qt, QBasicTimer, QPoint, QByteArray

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from nyst import projection
from nyst import gaze_loop
from nyst import stream
from nyst import face

import cv2

def current_time_sec():
	return int(round(time.time()))

class GazeTracker(QWidget):
	def __init__(self, camera, sensor_width_mm, sensor_height_mm, hdata_filename, vdata_filename, factors_filename):
		super().__init__()
		screen_mm = np.loadtxt("data/screen.txt")
		self.width_mm = screen_mm.item(0)
		self.height_mm = screen_mm.item(1)
		self.initUI()

		self.timer = QBasicTimer()
		self.timer.start(10, self)

		self.phase = 0.1
		self.count = 0

		self.x = int(self.width * self.phase)
		self.y = int(self.height * 0.5)

		self.state = "WAITING"

		self.recorded_x_phase = []
		self.recorded_x_face = []
		self.recorded_left_eye = []
		self.recorded_right_eye = []
		self.recorded_y_phase = []
		self.recorded_y_face = []
		self.recorded_vertical = []

		self.hdata_filename = hdata_filename
		self.vdata_filename = vdata_filename
		self.factors_filename = factors_filename

		gaze_loop.start(self, camera, sensor_width_mm, sensor_height_mm)

	def display(self, points):
		self.points = points
		self.update()

	# Callback for recorded data
	def add_data(self, left_phase_x, right_phase_x, v_phase, face_phase_x, face_phase_y):
		if (self.state == "MOVE_LEFT" or self.state == "MOVE_RIGHT") and self.count > 0:
			if self.phase > 0.15 and self.phase < 0.85:
				if len(self.recorded_x_face) > 0 and self.recorded_x_face[-1] == face_phase_x:
					return
				self.recorded_x_phase.append(self.phase)
				self.recorded_x_face.append(face_phase_x)
				self.recorded_left_eye.append(left_phase_x)
				self.recorded_right_eye.append(right_phase_x)
		
		if (self.state == "MOVE_DOWN" or self.state == "MOVE_UP") and self.count > 0:
			if self.phase > 0.15 and self.phase < 0.85:
				if len(self.recorded_y_face) > 0 and self.recorded_y_face[-1] == face_phase_y:
					return
				self.recorded_y_phase.append(self.phase)
				self.recorded_y_face.append(face_phase_y)
				self.recorded_vertical.append(v_phase)

	def millimeterToPixel(self,x,y):
		return (x / self.width_mm) * self.width, (y / self.height_mm) * self.height

	def initUI(self):      
		screen = QDesktopWidget().screenGeometry()

		self.width = screen.width()
		self.height = screen.height()
		print("Screen size: ", self.width, self.height, self.width_mm,  self.height_mm)
		self.point = -1, -1
		self.setGeometry(0, 0, self.width, self.height)

		self.setWindowFlags(Qt.FramelessWindowHint)
		self.show()

	def paintEvent(self, e):
		qp = QPainter()
		qp.begin(self)
		qp.setPen(QPen(Qt.red, 40, Qt.SolidLine))
		qp.drawEllipse(self.x,self.y,40,40)

		qp.end()

	def keyPressEvent(self, e):
		if e.key() == Qt.Key_F5:
			sys.exit(0)
		if self.state == "WAITING" and e.key() == Qt.Key_Space:
			self.state = "MOVE_RIGHT"
		if self.state == "WAITING_VERTICAL" and e.key() == Qt.Key_Space:
			self.state = "MOVE_DOWN"
		if self.state == "DONE" and e.key() == Qt.Key_Space:
			sys.exit(0)

	def timerEvent(self, event):
		'''handles timer event'''
		
		if event.timerId() != self.timer.timerId():
			return

		if self.state == "MOVE_LEFT":
			self.phase = self.phase - 0.01
			self.x = int(self.width * self.phase)
			if self.phase < 0.1:
				self.count = self.count + 1
				if self.count == 3:
					self.state = "WAITING_VERTICAL"
					self.count = 0
					self.x = int(self.width * 0.5)
					self.phase = 0.1
					self.y = int(self.height * self.phase)
				else:
					self.state = "MOVE_RIGHT"

		if self.state == "MOVE_RIGHT":
			self.phase = self.phase + 0.01
			self.x = int(self.width * self.phase)
			if self.phase >= 0.9:
				self.state = "MOVE_LEFT"

		if self.state == "MOVE_UP":
			self.phase = self.phase - 0.010
			self.y = int(self.height * self.phase)
			if self.phase < 0.1:
				self.count = self.count + 1
				if self.count == 3:
					self.done()
				else:
					self.state = "MOVE_DOWN"

		if self.state == "MOVE_DOWN":
			self.phase = self.phase + 0.010
			self.y = int(self.height * self.phase)
			if self.phase >= 0.9:
				self.state = "MOVE_UP"

		self.update()

	def done(self):
		self.x = -100
		self.y = -100
		self.calculate_parameters()
		self.state = "DONE"

	def closeEvent(self, event):
		gaze_loop.stop()
		sys.exit(0)

	def calculate_parameters(self):
		# Analyse Left Eye Phase + Face phase
		def fn(x, face_phase_factor, eye_phase_factor, a):
			return face_phase_factor*x[0] + eye_phase_factor*x[1] + a

		# left eye horizontal
		x = np.row_stack((self.recorded_x_face, self.recorded_left_eye))
		y = self.recorded_x_phase

		popt, _ = curve_fit(fn, x, y)
		face_phase_factor_left = popt[0]
		eye_phase_factor_left = popt[1]
		left_a = popt[2]

		# right eye horizontal
		x = np.row_stack((self.recorded_x_face, self.recorded_right_eye))
		y = self.recorded_x_phase

		popt, _ = curve_fit(fn, x, y)
		face_phase_factor_right = popt[0]
		eye_phase_factor_right = popt[1]
		right_a = popt[2]

		# vertical
		x = np.row_stack((self.recorded_y_face, self.recorded_vertical))
		y = self.recorded_y_phase

		popt, _ = curve_fit(fn, x, y)
		face_phase_factor_vertical = popt[0]
		eye_phase_factor_vertical = popt[1]
		vertical_a = popt[2]

		self.write_h_data_to_file(self.hdata_filename)
		self.write_v_data_to_file(self.vdata_filename)
		print("Calibration data written to: " + self.hdata_filename + ", " + self.vdata_filename)
		self.write_factors_to_file(self.factors_filename, face_phase_factor_left, eye_phase_factor_left, left_a, face_phase_factor_right, eye_phase_factor_right, right_a, face_phase_factor_vertical, eye_phase_factor_vertical, vertical_a)
		print("Calibration factors written to: " + self.factors_filename)

	def write_h_data_to_file(self, filename):
		with open(filename, "w+") as f:
			for i in range(len(self.recorded_x_phase)):
				line = "{};{};{};{}".format(self.recorded_x_phase[i], self.recorded_x_face[i], self.recorded_left_eye[i], self.recorded_right_eye[i])
				print(line, file=f)

	def write_v_data_to_file(self, filename):
		with open(filename, "w+") as f:
			for i in range(len(self.recorded_y_phase)):
				line = "{};{};{}".format(self.recorded_y_phase[i], self.recorded_y_face[i], self.recorded_vertical[i])
				print(line, file=f)

	def write_factors_to_file(self, filename, left_face, left_eye, left_a, right_face, right_eye, right_a, vertical_face, vertical_eye, vetrical_a):
		with open(filename, "w+") as f:
				line = "{};{};{}".format(left_face, left_eye, left_a)
				print(line, file=f)
				line = "{};{};{}".format(right_face, right_eye, right_a)
				print(line, file=f)
				line = "{};{};{}".format(vertical_face, vertical_eye, vetrical_a)
				print(line, file=f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--camera", help="camera number")
	parser.add_argument("-wi", "--width", help="sensor width in mm", type=float)
	parser.add_argument("-he", "--height", help="sensor height in mm", type=float)
	parser.add_argument("-dh", "--hdata", help="file name for horizontal calibration data")
	parser.add_argument("-dv", "--vdata", help="file name for vertical calibration data")
	parser.add_argument("-f", "--factors", help="file name for calibration factors")
	args, leftovers = parser.parse_known_args()
	app = QApplication(sys.argv)

	ex = GazeTracker(args.camera, args.width, args.height, args.hdata, args.vdata, args.factors)

	sys.exit(app.exec_())
