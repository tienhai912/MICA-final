import numpy as np
import scipy.io as sio
import os
import zipfile
from datetime import datetime
import logging
from PIL import Image
from scipy import sparse

from sklearn.metrics.pairwise import cosine_similarity

initial_1024_codebook = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\dictionary\\Caltech101_SIFT_Kmeans_1024.mat'
initial_matrix = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\matrix\\random_matrix.txt'
initial_codebook = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\matrix\\random_codebook.txt'
trained_codebook_file = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\codebook\\5_people_codebook.txt'
out_code_file = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\matrix\\out_code.txt'
pooling_file = 'pooling.txt'
llc_file = 'llc.txt'


def unzip_file(zip_file, unzip_folder):
  with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(unzip_folder)

def read_features_out(file_path):
  with open(file_path, 'r') as f:
    content = f.readlines()

  matrix = np.array([x.strip().split('\t') for x in content], dtype='float64')
  return matrix

def vector_distance(vector1, vector2):
  return numpy.linalg.norm(vector1-vector2)

def element_d(element_x, group_b):
  d = np.array([])
  for column in group_b.T:
    d.append(vector_distance(element_x, column))
  return d

def element_multiplication(element_d, element_c):
  return np.multiply(element_d, element_c)

# "bold 1" matrix
def identity_matrix(size):
  return np.identity(size)

def L2(x):
  return np.linalg.norm(x)

# Norm 2 of matrix according to input_axis
# input_axis = 0 -> Norm 2 of columns -> return vector length = row length
# input_axis = 1 -> Norm 2 of rows -> return vector length = column length
def L2_axis(x, input_axis):
  return np.linalg.norm(x, axis=input_axis)

def matrix_multiplication(matrix1, matrix2):
  return np.dot(matrix1, matrix2)

def random_vector(size):
  return np.random.random(size)

def random_matrix(row, col):
  return np.random.rand(row, col)

def gen_random_matrix():
  X = random_matrix(426, 5)
  np.savetxt(initial_matrix, X)
  B = random_matrix(426, 1024)
  np.savetxt(initial_codebook, B)

def load_initial_matrix():
  return np.loadtxt(initial_matrix)

def load_initial_codebook():
  return np.loadtxt(initial_codebook)

def load_initial_1024_codebook():
  temp = (sio.loadmat(initial_1024_codebook))['B']
  codebook = np.concatenate((temp, temp, temp, temp[:42]))
  return codebook

def remove_non_feature_column(feature_matrix):
  return np.delete(feature_matrix, np.s_[:11], axis=1)

def get_train_vector(input_folder):
  sorted_by_length = 'out_features_sorted_by_length.txt'
  sorted_by_length_zip = 'out_features_sorted_by_length.zip'

  unzip_file(os.path.join(input_folder, sorted_by_length_zip), input_folder)

  feature_matrix = read_features_out(os.path.join(input_folder, sorted_by_length))

  # feature_matrix = remove_non_feature_column(feature_matrix)

  os.remove(os.path.join(input_folder, sorted_by_length))

  return feature_matrix.T

def llc(B,X,knn): # LLC_coding_appr.m
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

  IDX = np.zeros((nframe, knn), dtype=int)

  for i in np.arange(nframe):
    d = D[i, :]
    index_d = np.argsort(d)
    IDX[i, :] = index_d[0:knn]

  II = np.identity(knn)
  Coeff = np.zeros((nframe, nbase))

  for i in np.arange(nframe):
    idx = IDX[i, :]
    z = B[idx, :] - np.tile(X[i, :], (knn, 1))
    C = np.dot(z, z.T)
    C = C + II*beta*np.trace(C)
    w = np.linalg.solve(C, np.ones((knn, 1)))
    w = w/np.sum(w)
    Coeff[i, idx] = w.T

  return Coeff

def LLC_pooling(B,X,pyramid,knn):
  X = X.T
  img_width = int(X[0, 2]/X[0, 8])
  img_height = int(X[0, 3]/X[0, 9])
  # X_mean = X.mean(axis=0)
  X1 = X[:, 2]
  Y = X[:, 3]

  X = remove_non_feature_column(X)
  X = X.T

  dSize = B.shape[1]
  nSmp = X.shape[1]

  idxBin = np.transpose(np.zeros(nSmp))

  # llc coding
  llc_codes = llc(np.transpose(B),np.transpose(X),knn)
  # im = Image.fromarray(llc_codes*255)
  # im.show()
  # print(llc_codes.shape)
  # print(np.count_nonzero(llc_codes))
  llc_codes = np.transpose(llc_codes)

  pLevels = len(pyramid) #spatial levels
  pBins = np.array(pyramid)*np.array(pyramid) #spatial bins on each level
  tBins = int(sum(pBins))

  beta = np.matrix(np.zeros((dSize,tBins)))
  bId = -1

  for i in range(pLevels):
    nBins = pBins[i]

    wUnit = img_width / pyramid[i]
    hUnit = img_height / pyramid[i]

    # find to which spatial bin each local descriptor belongs
    xBin = np.ceil(X1 / wUnit)
    yBin = np.ceil(Y / hUnit)
    idxBin = (yBin - 1)*pyramid[i] + xBin

    for j in range(int(nBins)):
      bId = bId + 1
      sidxBin = np.where(idxBin == j+1)[0]
      if len(sidxBin) == 0:
          continue
      beta[:,bId] = np.transpose(np.matrix(np.amax(llc_codes[:,sidxBin],axis=1)))

  # print(beta.shape)
  # print(np.cou nt_nonzero(beta))
  beta = np.transpose(np.transpose(beta).ravel())
  beta = beta/np.sqrt(np.sum(np.square(beta)))
  return beta

def run_pooling_LLC():
  input_root_folder = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\'
  # input_root_folder = '/home/dangmanhtruong95/NTHai/iDT_output/'

  # people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuan', 'Thuy', 'Tuyen']
  # kinects = ['Kinect_1', 'Kinect_2', 'Kinect_3', 'Kinect_4', 'Kinect_5']
  # people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuan']

  people = ['Giang']
  kinects = ['Kinect_1']

  knn = 5

  pyramid = [1, 2, 4]

  B = read_features_out(trained_codebook_file)
  im = Image.fromarray(B*255)
  im.show()
  A = []

  for person in people:
    for kinect in kinects:
      # folder_list = list_all_folders_in_a_directory(os.path.join(input_root_folder, person, kinect))
      folder_list = ['10_1', '10_2']
      for folder in folder_list:

        input_features_folder = os.path.join(input_root_folder, person, kinect, folder)
        feature_vector = get_train_vector(input_features_folder)

        out = LLC_pooling(B.T, feature_vector, pyramid, knn)
        # with open(os.path.join(input_features_folder, pooling_file),'wb') as f:
        #   np.savetxt(f, codebook, fmt='%7f', delimiter='\t')
        # print(B.shape)
        # print(remove_non_feature_column(feature_vector.T).shape)
        out = llc(B, remove_non_feature_column(feature_vector.T), 5)
        sp = sparse.csr_matrix(out)
        print(sp)
        with open(os.path.join(input_features_folder, llc_file),'wb') as f:
          np.savetxt(f, out, fmt='%7f', delimiter='\t')
        # A.append(out)

  # print(average_similarity(A[0], A[1]))
  # print(average_similarity(A[0], A[2]))
  # print(average_similarity(A[0], A[3]))
  # print(average_similarity(A[1], A[2]))
  # print(average_similarity(A[1], A[3]))
  # print(average_similarity(A[2], A[3]))

def average_similarity(matrix1, matrix2, THRESHOLD=0):
  # Check sum of all features
  sum_matrix1 = np.ravel(matrix1.sum(axis=1))
  sum_matrix2 = np.ravel(matrix2.sum(axis=1))

  # Take line numbers of data-rich features
  data_rich_feature_1 = [i for i in range(len(sum_matrix1)) if (sum_matrix1[i] > THRESHOLD) and (sum_matrix2[i] > THRESHOLD)]

  # Cosine similarity rows of matrix1 with rows of matrix2
  sim_matrix = cosine_similarity(matrix1,matrix2)

  # Take diagonal and remove NANs (if exist though there should not be one)
  diagonal = np.nan_to_num(sim_matrix.diagonal())

  # Use threshold ???????????????????????????????????????????????????????????????????????????????
  diagonal_threshold = [diagonal[i] for i in data_rich_feature_1]

  # Take sum distance and number of features
  sim_sum = np.sum(diagonal_threshold)
  sim_count = len(diagonal_threshold)
  return sim_sum/sim_count

def test():

  # train_codebook()
  # print(np.exp(-1))

  print('Test finished')

def test_llc():
  X = load_initial_matrix().T
  code_X = llc()

  with open(out_code_file,'wb') as f:
    np.savetxt(f, code_X, fmt='%7f', delimiter='\t')

  print(code_X)
  print(code_X.shape)
  print(X.shape)

if __name__ == '__main__':
  # gen_random_matrix()
  # test()
  # llc()
  # LLC_solution()
  # test_llc()
  run_pooling_LLC()
