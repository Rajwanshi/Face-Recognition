import numpy as np
import pandas as pd
import cv2 as cv

class PCA(object):
	
	def __init__(self, dim):
		self.dim = dim
		self.data = None
		self.originalShape = None
		self.finalShape = None
		self.cov_matrix = None
		self.verbose = False
		self.transformation_matrix = None

	def loadData(self, data):
		self.data = data
		self.originalShape = (self.data.shape[0], self.data.shape[1])
		self.finalShape = (self.data.shape[0], self.dim)
		self.cov_matrix = np.zeros((self.data.shape[1], self.data.shape[1]))
		self.transformation_matrix = np.zeros((self.data.shape[1], self.dim))
		print(data.shape)

	def _print(self, dtype, data):
		if self.verbose:
			print(dtype)
			for line in data:
				print(line)

	def transform(self, verbose=False):
		assert self.dim <= min(self.data.shape[0], self.data.shape[1]), "output dimensionality cannot be greater than original data dimensions"
		self.verbose = verbose
		self.col_mean = self.data.mean(axis = 0)
		self.data -= self.col_mean
		self.cov_matrix = np.dot(self.data.T, self.data)
		
		eig_val, eig_vec = np.linalg.eig(self.cov_matrix)
		eig_pairs = [(i.real,j.real) for i, j in zip(eig_val, eig_vec.T)]
		eig_pairs_sorted = sorted(eig_pairs, key = lambda k: k[0], reverse = True)
		eig_sum = np.sum(eig_val.real)
		self._print('Tuple of eigenvalues and eigenvectors', eig_pairs_sorted)
		
		contr = ["eigenvalue = {0} : contribution = {1:.2%}".format(pair[0],pair[0]/eig_sum) for pair in eig_pairs_sorted]
		self._print('Contribution of each eigenvalue', contr)
		
		for i in range(self.dim):
			self.transformation_matrix[:,i] = eig_pairs_sorted[i][1].T
		self._print('transformation_matrix', [self.transformation_matrix])
		return np.array(self.data.dot(self.transformation_matrix)), self.transformation_matrix

	def project(self, X):
		col_mean = X.mean(axis = 0)
		X -= col_mean
		return np.array(X.dot(self.transformation_matrix))

	def reconstruct(self, X):
		reconstructed = np.dot(X, self.transformation_matrix.T)
		print(reconstructed.shape, self.col_mean.shape)
		reconstructed += self.col_mean.values.reshape(1,self.col_mean.shape[0])
		return np.array(reconstructed)