import cv2 as cv

class FaceCrop(object):

    def __init__(self, ox, oy):
        self._img = None
        self._xml = "haarcascade_frontalface_default.xml"
        self._cascade = cv.CascadeClassifier(self._xml)
        self._minSize = None
        self._minFrame = None
        self.x = ox
        self.y = oy

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, newImg):
        self._img = newImg
        self._minSize = (self._img.shape[1], self._img.shape[0])
        self._minFrame = cv.resize(self._img, self._minSize)

    def identifyFaces(self):
        if self._img is not None:
            faces = self._cascade.detectMultiScale(self._minFrame)
            face_detected = []
            for f in faces:
                x, y, w, h = [ v for v in f ]
                w = self.x
                h = self.y
                img = self._img[y:y+h,x:x+w]
                face_detected.append(img)
            return face_detected
        else:
            return