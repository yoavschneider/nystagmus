import sys, random, argparse, time, os

from PyQt5.QtWidgets import QWidget, QApplication, QDesktopWidget
from PyQt5.QtGui import QPainter, QColor, QImage, QPen
from PyQt5.QtCore import Qt, QBasicTimer, QPoint, QByteArray

import numpy as np

from nyst import projection

import cv2
import cv2.aruco as aruco

def current_time_sec():
	return int(round(time.time()))

class ExtrinsicCalibrator(QWidget):
	def __init__(self, camera, k, dist):
		super().__init__()
		
		self.K = np.loadtxt(k)
		self.dist_factors = np.loadtxt(dist)

		screen = np.loadtxt("data/screen.txt")

		self.mm_width = screen[0]
		self.mm_height = screen[1]

		self.marker_size = 400
		self.last_time_found = current_time_sec()

		self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

		aruco1 = aruco.drawMarker(self.aruco_dict, 1, self.marker_size)
		aruco2 = aruco.drawMarker(self.aruco_dict, 2, self.marker_size)
		aruco3 = aruco.drawMarker(self.aruco_dict, 3, self.marker_size)

		self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

		# Mirror markers
		cv2.flip(aruco1, 1, aruco1)
		cv2.flip(aruco2, 1, aruco2)
		cv2.flip(aruco3, 1, aruco3)

		self.marker1 = QImage(aruco1.tobytes(), self.marker_size, self.marker_size, QImage.Format_Grayscale8)
		self.marker2 = QImage(aruco2.tobytes(), self.marker_size, self.marker_size, QImage.Format_Grayscale8)
		self.marker3 = QImage(aruco3.tobytes(), self.marker_size, self.marker_size, QImage.Format_Grayscale8)

		self.initUI()
		self.timer = QBasicTimer()
		self.timer.start(50, self)

		self.cap = cv2.VideoCapture(int(camera))

		self.edge_dist = 50

		self.found = []

		self.first = True

		x1,y1,z1 = self.get_reference_point_in_mm(self.edge_dist, self.edge_dist, 0)
		x2,y2,z2 = self.get_reference_point_in_mm(self.width - self.edge_dist, self.edge_dist, 0)
		x3,y3,z3 = self.get_reference_point_in_mm(self.edge_dist, self.height - self.edge_dist, 0)

		print ("******************************")
		print ("")
		print ("Reference points: ")
		print (x1,y1,z1)
		print (x2,y2,z2)
		print (x3,y3,z3)
		print ("")
		print ("******************************")

		model = np.matrix([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]])
		np.savetxt("data/ex_model.txt", model)

		print ("******************************")
		print ("")
		print ("K:")
		print (self.K)
		print ("")
		print ("******************************")

		self.count = 0
		self.done = False

	def get_reference_point_in_mm(self, x, y, z):
		x_mm = (x / self.width) * self.mm_width
		y_mm = (y / self.height) * self.mm_height
		return x_mm, y_mm, z

	def initUI(self):      
		screen = QDesktopWidget().screenGeometry()

		self.width = screen.width()
		self.height = screen.height()

		self.setGeometry(0, 0, self.width, self.height)

		self.setWindowFlags(Qt.FramelessWindowHint)
		self.show()

	def findMarkers(self):
		_, frame = self.cap.read()

		frame = cv2.undistort(frame, self.K, self.dist_factors, None, self.K)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict)

		self.found = ids if ids is not None else []

		height, width, _ = frame.shape

		# All markers found
		if not self.done and 1 in self.found and 2 in self.found and 3 in self.found:
			if self.first:
				self.last_time_found = current_time_sec()
				self.first = False
				return

			ids = ids.flatten().tolist()
			index1 = ids.index(1)
			index2 = ids.index(2)
			index3 = ids.index(3)
			x1,y1 = corners[index1][0][1][0], corners[index1][0][1][1]
			x2,y2 = corners[index2][0][0][0], corners[index2][0][0][1]
			x3,y3 = corners[index3][0][2][0], corners[index3][0][2][1]

			# Make sure there is a delay after points were successfuly saved
			if current_time_sec() - self.last_time_found > 5:
				markers = np.array([[x1,y1],[x2,y2],[x3,y3]])
				self.subpix_corners(markers, gray)
				h,w, _ = frame.shape
				cv2.imwrite('EXCALIB' + str(self.count) + ".jpg", frame)

				cv2.rectangle(frame, (0,0), (w,h), (0,0,255), h)
				resized = cv2.resize(frame, (400, int (400 * (height / width))))
				cv2.imshow("img", resized)
				cv2.waitKey(10)
				time.sleep(1)
				self.last_time_found = current_time_sec()
				self.count = self.count + 1
				np.savetxt("data/ex_input" + str(self.count) + ".txt", markers)
				print("Saved: data/ex_input" + str(self.count) + ".txt")
				print()
				if self.count == 3:
					time.sleep(1)
					self.done = True
					# Do calibration with TAKASHI et. al algorithm
					os.system("./demo")
					
					# Visualize result, Sensor sized is not used
					proj = projection.Projection(self.width, 1, self.height, 1)
					proj.visualize()
					sys.exit(0)

			# Visualize points found in camera capture window
			cv2.circle(frame, (x1,y1), 2, (0,0,255), 2)
			cv2.circle(frame, (x2,y2), 2, (0,0,255), 2)
			cv2.circle(frame, (x3,y3), 2, (0,0,255), 2)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			sys.exit(0)

		resized = cv2.resize(frame, (400, int (400 * (height / width))))
		cv2.imshow("img", resized)

	def subpix_corners(self, corners, gray):
		#print(corners)
		cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),self.criteria)
		print(corners)

	def paintEvent(self, e):
		qp = QPainter()
		qp.begin(self)
		
		qp.drawImage(self.edge_dist,self.edge_dist,self.marker1)
		qp.drawImage(self.width - self.marker_size - self.edge_dist,self.edge_dist,self.marker2)
		qp.drawImage(self.edge_dist,self.height - self.marker_size - self.edge_dist,self.marker3)

		pen = QPen(Qt.red, 5, Qt.SolidLine)
		qp.setPen(pen)

		if 1 in self.found and 2 in self.found and 3 in self.found:
			pen = QPen(Qt.green, 5, Qt.SolidLine)
			qp.setPen(pen)

		qp.end()
	
	def timerEvent(self, event):
		'''handles timer event'''
		
		if event.timerId() == self.timer.timerId() and not self.done:
			self.findMarkers()
			self.update()

	def closeEvent(self, event):
		sys.exit(0)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--camera", help="camera port number")
	parser.add_argument("-m", "--matrix", help="camera matrix file name", default="data/camera-matrix.txt")
	parser.add_argument("-d", "--distortion", help="camera distortion factors file name", default="data/distortion-factors.txt")
	args, leftovers = parser.parse_known_args()
	app = QApplication(sys.argv)

	ex = ExtrinsicCalibrator(args.camera, args.matrix, args.distortion)

	sys.exit(app.exec_())
