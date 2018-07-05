import cv2 as cv
from faceCrop import FaceCrop
import numpy as np

class Window(object):

	def __init__(self, windowName, vc):
		self._vc = vc
		self._fps = 30
		self._size = (int(self._vc.get(cv.CAP_PROP_FRAME_WIDTH)), int(self._vc.get(cv.CAP_PROP_FRAME_HEIGHT)))
		self._frames_to_capture = 2*self._fps - 1
		self._capture = False
		self.windowName = windowName

	@property
	def frames(self):
		return self._frames_to_capture

	@frames.setter
	def frames(self, newVal):
		self._frames_to_capture = newVal

	@property
	def capture(self):
		return self._capture

	def createWindow(self):
		cv.namedWindow(self.windowName)

	def handleKeyboard(self, key):
		if key == 115 and not self._capture:
			self._capture = not self._capture
	
	@property
	def size(self):
		return self._size

	def getFrame(self):
		_, frame = self._vc.read()
		return frame

	def delWindow(self):
		cv.destroyWindow(self.windowName)

class CaptureFrames(object):

	def __init__(self):
		self._windowname = 'FaceRecog-DataCapture'
		self._window = Window(self._windowname,
										cv.VideoCapture(0))
		self._frame = None
		self._fc = FaceCrop(210, 210)
		self.count = 0

	def writeFace(self):
		while self._window.frames > 0:
			# print(capture)
			self._window.handleKeyboard(cv.waitKey(1))
			self._frame = self._window.getFrame()
			# self._frame = cv.resize(self._frame, (0,0), fx=0.5, fy=0.5) 
			self.mirrored_frame = np.fliplr(self._frame).copy()
			if self._window.capture:
				self._fc.img = self._frame
				faces = self._fc.identifyFaces()
				if len(faces) > 1:
					print("Only one person required in the frame")
				elif len(faces) == 0:
					print("Please enter the frame")
				else:
					cv.imwrite('verification_1/'+str(self.count) + '.jpg', faces[0])
					self._window.frames -= 1
					self.count += 1
					print(self._window.frames)
			# self._window.frames -= 1
			cv.line(self.mirrored_frame, (self._window.size[0] // 3, 0), (self._window.size[0] // 3, self._window.size[1]), (255, 255, 255), 1, 1)
			cv.line(self.mirrored_frame, (2*self._window.size[0] // 3, 0), (2*self._window.size[0] // 3, self._window.size[1]), (255, 255, 255), 1, 1)
			cv.line(self.mirrored_frame, (0,self._window.size[1] // 5), (self._window.size[0], self._window.size[1] // 5), (255, 255, 255), 1, 1)
			cv.line(self.mirrored_frame, (0,4*self._window.size[1] // 5), (self._window.size[0] , 4*self._window.size[1] // 5), (255, 255, 255), 1, 1)
			
			cv.imshow(self._windowname, self.mirrored_frame)
		self._window.delWindow()

if __name__ == "__main__":
	CaptureFrames().writeFace()
