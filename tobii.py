import tobii_research as tr

import time, sys, argparse
from random import shuffle

from PyQt4.QtGui import QWidget, QDesktopWidget, QApplication, QPainter, QPen, QFont
from PyQt4.QtCore import Qt, QBasicTimer

def current_time_sec():
	return int(round(time.time()))

class TobiiCalibrator(QWidget):
	def __init__(self, redo_calibration):
		super(TobiiCalibrator, self).__init__()

		self.timer = QBasicTimer()
		self.timer.start(50, self)
		self.step = 'not_started'
		self.initUI()

		self.redo_calibration = redo_calibration

		self.points = []
		for x in (0.1,0.5,0.9):
			for y in (0.1,0.5,0.9):
				self.points.append((x,y))

		shuffle(self.points)

		self.x = -1
		self.y = -1

		found_eyetrackers = tr.find_all_eyetrackers()
		self.eyetracker = found_eyetrackers[0]
		print("Address: " + self.eyetracker.address)
		print("Model: " + self.eyetracker.model)
		print("Name (It's OK if this is empty): " + self.eyetracker.device_name)
		print("Serial number: " + self.eyetracker.serial_number)

		# Coordinates of the display area in user coordinate system
		display_area = self.eyetracker.get_display_area()
		print(display_area.top_left)
		print(display_area.top_right)
		print(display_area.bottom_left)
		print(display_area.bottom_right)

	def calibrate_start(self):
		self.calibration = tr.ScreenBasedCalibration(self.eyetracker)
		self.calibration.enter_calibration_mode()

	def show_next(self):
		if len(self.points) == 0:
			return True

		x,y = self.points[len(self.points) - 1]
		x,y = int(x  * self.width), int(y * self.height)
		self.show_point(x,y)

		return False

	def calibrate(self):
		x,y = self.points[len(self.points) - 1]

		if self.calibration.collect_data(x, y) != tr.CALIBRATION_STATUS_SUCCESS:
			self.calibration.collect_data(x, y)
		
		self.points.pop()

	def calibrate_done(self):
		calibration_result = self.calibration.compute_and_apply()
		print("Compute and apply returned {0} and collected at {1} points.".format(calibration_result.status, len(calibration_result.calibration_points)))
		self.calibration.leave_calibration_mode()

	def show_point(self,x,y):
		self.x = x
		self.y = y
		self.update()

	def paintEvent(self, e):
		qp = QPainter()
		qp.begin(self)

		if self.step == 'waiting_to_start':
			qp.setPen(QPen(Qt.red, 20, Qt.SolidLine))
			qp.setFont(QFont('Decorative', 20))

		if self.step == 'calibration_started':
			qp.setPen(QPen(Qt.red, 35, Qt.SolidLine))
			qp.drawEllipse(self.x,self.y,40,40)
			
		if self.step == 'show_gaze':
			qp.setPen(QPen(Qt.red, 10, Qt.SolidLine))
			qp.drawEllipse(self.x,self.y,10,10)
			
		qp.end()
	
	def timerEvent(self, event):
		'''handles timer event'''
		
		if event.timerId() == self.timer.timerId():
			if self.step == 'not_started' and not self.redo_calibration:
				try:
					with open("data/tobii_calibration_data", "rb") as f:
						calibration_data = f.read()
						if len(calibration_data) > 0:
							self.eyetracker.apply_calibration_data(calibration_data)
							print("not_started => show_gaze")
							print("Subscribing to gaze data for eye tracker with serial number {0}.".format(self.eyetracker.serial_number))
							self.eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True)
							self.step = 'show_gaze'
							return
				except IOError:
					print("not_started => waiting_to_start")
					self.step = 'waiting_to_start'
					self.timestamp = current_time_sec()
			elif self.step == 'not_started':
					print("not_started => waiting_to_start")
					self.step = 'waiting_to_start'
					self.timestamp = current_time_sec()

			if self.step == 'waiting_to_start':
				if current_time_sec() - self.timestamp > 1:
					print("waiting_to_start => calibration_started")
					self.step = 'calibration_started'
					self.timestamp = current_time_sec()
					self.calibrate_start()
					self.show_next()
				else:
					pass

			if self.step == 'calibration_started':
				if current_time_sec() - self.timestamp > 3:
					self.calibrate()
					self.timestamp = current_time_sec()
					done = self.show_next()
					if done:
						print("calibration_started => calibration_done")
						self.calibrate_done()
						self.step = 'calibration_done'

			if self.step == 'calibration_done':
				with open("data/tobii_calibration_data", "wb") as f:
					calibration_data = self.eyetracker.retrieve_calibration_data()
					if calibration_data is not None:
						print("Saving calibration to file for eye tracker with serial number {0}.".format(self.eyetracker.serial_number))
						f.write(calibration_data)
				
				print("Subscribing to gaze data for eye tracker with serial number {0}.".format(self.eyetracker.serial_number))
				self.eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback, as_dictionary=True)
				print("calibration_done => show_gaze")
				self.step = 'show_gaze'

			self.update()


	def gaze_data_callback(self, gaze_data):
		self.gaze_data = gaze_data
		x, y = gaze_data['left_gaze_point_on_display_area']
		print(x,y)
		self.x, self.y = int(x * self.width), int(y * self.height)
		self.update()

	def closeEvent(self, event):
		sys.exit(0)
	def initUI(self):      
		screen = QDesktopWidget().screenGeometry()

		self.width = screen.width()
		self.height = screen.height()

		self.setGeometry(0, 0, self.width, self.height)

		self.setWindowFlags(Qt.FramelessWindowHint)
		self.show()

	def keyPressEvent(self, e):
		if e.key() == Qt.Key_F5:
			sys.exit(0)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--redo", help="redo calibration even if exists", action='store_true')
	args, leftovers = parser.parse_known_args()

	app = QApplication(sys.argv)
	a = TobiiCalibrator(args.redo)
	sys.exit(app.exec_())