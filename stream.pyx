# based on: https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/


# import the necessary packages
from threading import Thread
import pygame
import cv2
import pygame.camera
import time

class WebcamVideoStream:
	def __init__(self, src=0, frame_rate = 30, width=640, height=480, format="RGB", bright=100):
		pygame.camera.init()
		cameras = pygame.camera.list_cameras()
		
		print("Available Cameras:")
		for cam in cameras:
			print(cam)

		print("Using /dev/video" + str(src))

		self.cam = pygame.camera.Camera("/dev/video" + str(src), (width, height), format)
		self.cam.start()
		time.sleep(0.1)

		img = self.cam.get_image()
		self.fresh = True
		color_image = cv2.transpose(pygame.surfarray.pixels3d(img))
		color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
		self.frame = color_image

		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		self.start_time = time.time()
		self.frame_count = 0

		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			img = self.cam.get_image()
			color_image = cv2.transpose(pygame.surfarray.pixels3d(img))
			color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
			self.frame = color_image
			self.fresh = True

			# print frame rate
			if (self.frame_count % 100 == 0):
				print("stream rate: ")
				print(self.frame_count / (time.time() - self.start_time))
				self.start_time = time.time()
				self.frame_count = 0

			self.frame_count = self.frame_count + 1

	def read(self):
		# return the frame most recently read
		if self.fresh:
			self.fresh = False
			return self.frame, True
		else:
			return self.frame, False

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True