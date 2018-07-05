from pca import PCA
import numpy as np
import pandas as pd
import cv2 as cv
import os
from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA

class FileReader(object):

	def __init__(self,src_root, dest_root):
		self.src_root = src_root
		self.dest_root = dest_root
		self.src = []

	def getPaths(self):
		for direc, folds, files in os.walk(self.src_root):
			self.src = [os.path.join(self.src_root, file) for file in files]
		return self.src


class ImgHandler(object):

	def __init__(self, filepaths):
		self.filepaths = filepaths
		self.data_matrix = []
	
	def transform(self):
		for file in self.filepaths:
			self.data_matrix.append(cv.imread(file,cv.IMREAD_GRAYSCALE).flatten())
		self.data_matrix = np.array(self.data_matrix)
		# print(self.data_matrix)

	def constructData(self):
		self.transform()
		self.df = pd.DataFrame(self.data_matrix.T, index=None, columns=None)
		return self.df

class EigenFaces(object):
	
	def __init__(self, path='face_samples/', dest='eigenfaces/'):
		self.reader = FileReader(path,dest)
		self.filepaths = self.reader.getPaths()
		self.imgHandler = ImgHandler(self.filepaths)
		self.res = 210

	def run(self):
		data = self.imgHandler.constructData()
		# print(data.head())
		# pca = PCA(n_components=4, svd_solver='full')
		# pca.fit(data) 
		# data_PCA = pca.transform(data)
		# print(pca.explained_variance_ratio_)
		p = PCA(3)
		p.loadData(data.copy())
		data_PCA, transformation_mat = p.transform()
		reconstruct = p.reconstruct(data_PCA)
		cv.imwrite('eigenfaces/rec.jpg',np.array(reconstruct)[:,0].reshape(self.res,self.res))
		cv.imwrite('eigenfaces/img1.jpg',np.array(data_PCA)[:,0].reshape(self.res,self.res))
		cv.imwrite('eigenfaces/img2.jpg',np.array(data_PCA)[:,1].reshape(self.res,self.res))
		cv.imwrite('eigenfaces/img3.jpg',np.array(data_PCA)[:,2].reshape(self.res,self.res))
		# cv.imwrite('eigenfaces/img4.jpg',np.array(data_PCA)[:,3].reshape(self.res,self.res))
		
		# mean_init = data_PCA.mean(axis=1)

		# self.__init__('verification_2','eigenfaces')
		# data_test = self.imgHandler.constructData()
		# data_test_PCA = p.project(data_test)
		# mean_test = np.array(data_test_PCA.mean(axis=1))
		# print(mean_init)
		# print(mean_test)
		# dist = np.linalg.norm(data_PCA[:,0]-data_test_PCA[:,0])
		# print(dist)
		# cv.imwrite('eigenfaces/rec.jpg',np.array(reconstruct)[:,0].reshape(self.res,self.res))
		# cv.imwrite('eigenfaces/img1.jpg',np.array(data_PCA)[:,0].reshape(self.res,self.res))
		# cv.imwrite('eigenfaces/img2.jpg',np.array(data_PCA)[:,1].reshape(self.res,self.res))
		# cv.imwrite('eigenfaces/img3.jpg',np.array(data_PCA)[:,2].reshape(self.res,self.res))
		# cv.imwrite('eigenfaces/img4.jpg',np.array(data_PCA)[:,3].reshape(self.res,self.res))
		cv.waitKey()

if __name__ == "__main__":
	EigenFaces().run()