import numpy as np
import scipy.io as sio

initial_1024_codebook = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\dictionary\\Caltech101_SIFT_Kmeans_1024.mat'
initial_matrix = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\matrix\\random_matrix.txt'
initial_codebook = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\matrix\\random_codebook.txt'
trained_codebook_file = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\codebook\\5_people_codebook.txt'

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

def llc(): # LLC_coding_appr.m
  # B: Mxd
  # X: Nxd
  B = load_initial_codebook().T
  # M = 1024
  nbase = B.shape[0]
  X = load_initial_matrix().T
  # N = 5
  nframe = X.shape[0]
  knn = 5
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

def LLC_solution():
  code_lambda = 0.0001
  code_sigma = 1
  epsilon = 0.000001

  B = load_initial_codebook()
  M = B.shape[1]
  X = load_initial_matrix()
  N = X.shape[1]
  d = np.zeros((N, M))
  for i in np.arange(N):
    xi = X[:, i]
    di = np.exp(L2_axis((xi - B.T).T, 0) / code_sigma)
    d[i,:] = di
    # print(d)
    # print(d.shape)

    # Normalize
    # NOTE: range of d: (epsilon,1]; epsilon = 10^-6
    # d = epsilon + (1 - epsilon)*(d - d.min()) / (d.max() - d.min())
  # start test
  # d = NxM
  # d = np.exp(L2((X - B.T).T, 0) / code_sigma)

  # d = epsilon + (1 - epsilon)*(d - d.min()) / (d.max() - d.min())
  Ci = (B - np.identity(xi.shape[1])*(xi.T)) * (B - np.identity(xi.shape[1])).T
  ci2 = np.linalg.solve((Ci - code_lambda * np.diag(d)), np.identity(Ci.shape))
  ci = ci2/(np.identity().T*ci2)

# Original code to calculate d in 'for i in np.arange(N):'
def train_codebook_cal_d(X, B, code_sigma, i):
  M = B.shape[1]
  d = np.zeros((1, M))
  for j in np.arange(M):
    d[:, j] = np.exp(-1) * L2(X[:, i] - B[:, j]) / code_sigma * -1
  return d

# least-squares solution to a linear matrix equation
# b = Ax (b,A,x are matrices)
def least_square(b, A):
  return np.linalg.lstsq(A,b)[0]

def least_square_example():
  A = np.array([[6, 1], [1, 6]])
  b = np.array([[9], [4]])
  print(np.linalg.lstsq(A,b)[0])
  # result = [[10/7], [3/7]]
  print(np.array([[10/7], [3/7]]))

def train_codebook():
  # In applying our algorithm, the two related
  # parameters λ and σ in Eq.(4) and Eq.(8) were carefully selected such that the cardinality, i.e. the length of id in line 8
  # of Alg.4.1, could match the number of neighbors used during classification
  code_lambda = 500
  code_sigma = 100
  B = load_initial_codebook()
  M = B.shape[1]
  X = load_initial_matrix()
  N = X.shape[1]
  for i in np.arange(N):
    xi = X[:, i]
    # train_codebook_cal_d(X, B, code_sigma, i)
    # d = np.exp(-1) * L2_axis((X[:, i] - B.T).T, 0) / code_sigma * -1

    # {locality constraint parameter}
    d = np.exp(L2_axis((xi - B.T).T, 0) / code_sigma)

    # Normalize
    # NOTE: range of d: (epsilon,1]; epsilon = 10^-6
    epsilon = 0.000001
    d = epsilon + (1 - epsilon)*(d - d.min()) / (d.max() - d.min())
    # print('-----------')
    # print(len(d))
    # print(d

    # {coding}
    ci = np.zeros((1, M))
    # {remove bias}
    id = [i for i in np.arange(M) if np.absolute(ci[i, 0]) > 0.01] # or ci[0, i] if ci is row vector
    Bi = [B[:, i] for i in id]

    ci2 = 0 #

    # {update basis}
    Bi2 = -2*ci2*(xi - Bi*ci2)
    muy = np.sqrt(1/i)
    Bi = Bi - muy*Bi/()
    print(Bi.shape)
    print(Bi2.shape)

def test():

  train_codebook()
  # print(np.exp(-1))

  print('Test finished')

# ------------ Sum pooling


if __name__ == '__main__':
  # gen_random_matrix()
  # test()
  # llc()
  # LLC_solution()
  test_code = read_features_out(trained_codebook_file)
  print(test_code.shape)
