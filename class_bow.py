import numpy as np
import os
import scipy.cluster.vq as vq
import zipfile
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import code_support as support
from datetime import datetime
import logging
from class_llc import LLC
from sklearn.metrics.pairwise import pairwise_distances

class BOW(LLC):


  def calculate_llc(self, B, X): # LLC_coding_appr.m
    # B: Mxd
    # X: Nxd
    ## B = read_features_out(trained_codebook_file)
    # M = 1024
    nbase = B.shape[0]
    ## X = load_initial_matrix().T
    # N = 5
    nframe = X.shape[0]
    ## knn = 5
    beta = 0.0001

    XX = np.sum(np.multiply(X, X), axis=1, keepdims=True)
    BB = np.sum(np.multiply(B, B), axis=1, keepdims=True)

    D = np.tile(XX, (1, nbase)) - 2*np.dot(X, B.T) + np.tile(BB.T, (nframe, 1))

    IDX = np.zeros((nframe, self.knn), dtype=int)

    for i in np.arange(nframe):
      d = D[i, :]
      index_d = np.argsort(d)
      IDX[i, :] = index_d[0:self.knn]

    II = np.identity(self.knn)
    Coeff = np.zeros((nframe, nbase))

    for i in np.arange(nframe):
      idx = IDX[i, :]
      z = B[idx, :] - np.tile(X[i, :], (self.knn, 1))
      C = np.dot(z, z.T)
      C = C + II*beta*np.trace(C)
      w = np.linalg.solve(C, np.ones((self.knn, 1)))
      w = w/np.sum(w)
      Coeff[i, idx] = w.T

    return Coeff

def test():
  X = np.array([[1, 1], [2, 2]])
  B = np.array([[3, 3], [4, 4], [5, 5]])

  A = pairwise_distances(X, B)
  print(A)
  print(A.shape)

if __name__ == '__main__':

  # first_llc = BOW()
  test()
