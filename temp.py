import cv2 as cv
import numpy as np
img = cv.imread('face_samples/4.jpg',cv.IMREAD_GRAYSCALE)
cv.imshow('img',img)
cv.waitKey()
