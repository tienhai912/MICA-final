import numpy as np
from datetime import datetime
import logging
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle


# input_root_folder = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\'
input_root_folder = '/home/dangmanhtruong95/NTHai/iDT_output/'

code_vector_folder_name = 'code_vector'
svm_test_data_folder_name = 'svm_test_data'
train_svm_folder_name = 'trained_svm'

out_feature_file = 'out_features.txt'
output_zip_file = 'output.zip'

for handler in logging.root.handlers[:]:
  logging.root.removeHandler(handler)

logging.basicConfig(filename='svm_20190502.log', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('Hai')

def list_all_folders_in_a_directory(directory):
  folder_list = (folder_name for folder_name in os.listdir(directory)
    if os.path.isdir(os.path.join(directory, folder_name)))
  return folder_list
  pass

def unzip_file(zip_file, unzip_folder):
  with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(unzip_folder)

def read_features_out(file_path):
  with open(file_path, 'r') as f:
    content = f.readlines()

  matrix = np.array([x.strip().split('\t') for x in content], dtype='float64')
  return matrix

def save_features_out(matrix, file_path):
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, 'wb') as f:
    np.savetxt(f, matrix, fmt='%7f', delimiter='\t')

def element_multiplication(element_d, element_c):
  return np.multiply(element_d, element_c)

# "bold 1" matrix
def identity_matrix(size):
  return np.identity(size)

def remove_non_origin_feature_column(feature_matrix):
  return np.delete(feature_matrix, np.s_[:10], axis=1)

# (2, 3) + (2, 5) = (2, 8)
def merge_two_matrices_row_by_row(matrix1, matrix2):
  return np.concatenate((matrix1, matrix2), axis=1)

# (2, 3) + (4, 3) = (6, 3)
def merge_two_matrices_col_by_col(matrix1, matrix2):
  return np.concatenate((matrix1, matrix2), axis=0)

def save_model(model, file_path):
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, 'wb') as f:
    pickle.dump(model, f)

def load_model(file_path):
  with open(file_path, 'rb') as f:
    model = pickle.load(f)

  return model

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

# Get iDT output without first 10 columns (only take trajectories, HOG, HOF, MBH)
def get_origin_trajectories(input_folder):
  unzip_file(os.path.join(input_folder, output_zip_file), input_folder)

  feature_matrix = read_features_out(os.path.join(input_folder, out_feature_file))

  feature_matrix = remove_non_origin_feature_column(feature_matrix)

  os.remove(os.path.join(input_folder, out_feature_file))

  return feature_matrix

# Vectorize iDT code and save result vectors with their corresponding label as txt
def gen_vector_data(kinect, codebook_kinect, test_person, knn):
  people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuy', 'Tuyen']

  codebook_file = os.path.join(input_root_folder, 'codebook', codebook_kinect + '_' + test_person + '_codebook.txt')
  B = read_features_out(codebook_file)

  for person in people:
    temp_X_sum = np.empty((0, 1024))
    temp_X_max = np.empty((0, 1024))
    temp_y = np.empty((0, 1))
    folder_list = list_all_folders_in_a_directory(os.path.join(input_root_folder, person, kinect))
    # folder_list = ['10_1', '10_2']
    for folder in folder_list:

      input_features_folder = os.path.join(input_root_folder, person, kinect, folder)
      feature_vector = get_origin_trajectories(input_features_folder)

      code_vector = llc(B, feature_vector, knn)
      # Sum pooling and max pooling
      code_vector_sum = np.sum(code_vector, axis=0, keepdims=True)
      code_vector_max = np.amax(code_vector, axis=0, keepdims=True)
      # Norm 2?
      code_vector_sum = code_vector_sum/np.sqrt(np.sum(np.square(code_vector_sum)))
      code_vector_max = code_vector_max/np.sqrt(np.sum(np.square(code_vector_max)))
      # label
      label_vector = np.array([[float(folder.split('_')[0])]])

      temp_X_sum = merge_two_matrices_col_by_col(temp_X_sum, code_vector_sum)
      temp_X_max = merge_two_matrices_col_by_col(temp_X_max, code_vector_max)
      temp_y = merge_two_matrices_col_by_col(temp_y, label_vector)

    save_vector_folder = os.path.join(input_root_folder, code_vector_folder_name)

    # kinect = kinect of input data
    # codebook_kinect = kinect of codebook
    prefix_vector_name = kinect + '_to_' + codebook_kinect + '_' + person + '_' + test_person +'_'

    logger.info(prefix_vector_name)
    logger.info('temp_X_sum: ' + str(temp_X_sum.shape))
    logger.info('temp_X_max: ' + str(temp_X_max.shape))
    logger.info('temp_y: ' + str(temp_y.shape))

    save_features_out(temp_X_sum, os.path.join(save_vector_folder, prefix_vector_name + 'X_sum.txt'))
    save_features_out(temp_X_max, os.path.join(save_vector_folder, prefix_vector_name + 'X_max.txt'))
    save_features_out(temp_y, os.path.join(save_vector_folder, prefix_vector_name + 'y.txt'))

# Run gen_vector_data on a subset of the input data
def run_gen_vector_data(kinects, try_leave_out_people, knn):
  logger.info('Start run gen vector data')
  # people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuy', 'Tuyen']
  # # kinects = ['Kinect_1', 'Kinect_2', 'Kinect_3', 'Kinect_4', 'Kinect_5']

  # try_leave_out_people = ['Thuy']

  # # people = ['Giang']
  # kinects = ['Kinect_4', 'Kinect_5']

  for kinect in kinects:
    for test_person in try_leave_out_people:
      gen_vector_data(kinect, kinect, test_person, knn)

  logger.info('Finished run gen vector data')

# Read vector files (output of run_gen_vector_data)
def read_vector_data(kinect, codebook_kinect, person, test_person):
  save_vector_folder = os.path.join(input_root_folder, code_vector_folder_name)
  prefix_vector_name = kinect + '_to_' + codebook_kinect + '_' + person + '_' + test_person +'_'

  temp_X_sum = read_features_out(os.path.join(save_vector_folder, prefix_vector_name + 'X_sum.txt'))
  temp_X_max = read_features_out(os.path.join(save_vector_folder, prefix_vector_name + 'X_max.txt'))
  temp_y = read_features_out(os.path.join(save_vector_folder, prefix_vector_name + 'y.txt'))

  return temp_X_sum, temp_X_max, temp_y

# Divide vector files into train and test sets
# Train through a SVC
# Save predictions and test label
def save_data(kinects, try_leave_out_people):
  logger.info('Start save data')
  svm_test_data_folder = os.path.join(input_root_folder, svm_test_data_folder_name)

  people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuy', 'Tuyen']

  for kinect in kinects:
    for test_person in try_leave_out_people:
      data = {}
      data['X_train_sum'] = np.empty((0, 1024))
      data['X_test_sum'] = np.empty((0, 1024))
      data['X_train_max'] = np.empty((0, 1024))
      data['X_test_max'] = np.empty((0, 1024))
      data['y_train'] = np.empty((0, 1))
      data['y_test'] = np.empty((0, 1))

      for person in people:
        temp_X_sum, temp_X_max, temp_y = read_vector_data(kinect, kinect, person, test_person)
        if (person!=test_person):
          data['X_train_sum'] = merge_two_matrices_col_by_col(data['X_train_sum'], temp_X_sum)
          data['X_train_max'] = merge_two_matrices_col_by_col(data['X_train_max'], temp_X_max)
          data['y_train'] = merge_two_matrices_col_by_col(data['y_train'], temp_y)
        else:
          data['X_test_sum'] = merge_two_matrices_col_by_col(data['X_test_sum'], temp_X_sum)
          data['X_test_max'] = merge_two_matrices_col_by_col(data['X_test_max'], temp_X_max)
          data['y_test'] = merge_two_matrices_col_by_col(data['y_test'], temp_y)

      # Create and train
      sum_svclassifier = SVC(kernel='linear')
      max_svclassifier = SVC(kernel='linear')
      sum_svclassifier.fit(data['X_train_sum'], data['y_train'].ravel())
      max_svclassifier.fit(data['X_train_max'], data['y_train'].ravel())

      # kinect = kinect train
      # test_person = leave one out person
      svc_prefix_name = kinect + '_' + test_person + '_'
      save_model(sum_svclassifier, os.path.join(input_root_folder, train_svm_folder_name, svc_prefix_name + 'sum.pkl'))
      save_model(max_svclassifier, os.path.join(input_root_folder, train_svm_folder_name, svc_prefix_name + 'max.pkl'))

      # folder to save data of each leave one out test
      # fisrt kinect = kinect of input data
      # second kinect = kinect of codebook
      # test_person = the person got left out to test (the remaining people are used to train)
      save_test_folder = os.path.join(svm_test_data_folder, kinect + '_to_' + kinect + '_' + test_person)

      logger.info(save_test_folder)
      logger.info('X_test_sum: ' + str(data['X_test_sum'].shape))
      logger.info('X_test_max: ' + str(data['X_test_max'].shape))
      logger.info('y_test: ' + str(data['y_test'].shape))

      save_features_out(data['X_test_sum'], os.path.join(save_test_folder, 'X_test_sum.txt'))
      save_features_out(data['X_test_max'], os.path.join(save_test_folder, 'X_test_max.txt'))
      save_features_out(data['y_test'], os.path.join(save_test_folder, 'y_test.txt'))

  logger.info('Finished save data')

def predict(test_person, test_kinect, train_kinect):
  sum_svc = load_model(os.path.join(input_root_folder, train_svm_folder_name, train_kinect + '_' + test_person + '_' + 'sum.pkl'))
  max_svc = load_model(os.path.join(input_root_folder, train_svm_folder_name, train_kinect + '_' + test_person + '_' + 'max.pkl'))

  test_data_folder = os.path.join(input_root_folder, svm_test_data_folder_name, test_kinect + '_to_' + test_kinect + '_' + test_person)
  X_test_sum = read_features_out(os.path.join(test_data_folder, 'X_test_sum.txt'))
  X_test_max = read_features_out(os.path.join(test_data_folder, 'X_test_max.txt'))
  y_test = read_features_out(os.path.join(test_data_folder, 'y_test.txt'))

  y_pred_sum = sum_svc.predict(X_test_sum)
  y_pred_max = max_svc.predict(X_test_max)

  percen_sum = np.sum(1 for i, j in zip(y_pred_sum, y_test) if i == j)/len(y_test)
  percen_max = np.sum(1 for i, j in zip(y_pred_max, y_test) if i == j)/len(y_test)

  return percen_sum, percen_max

def run_predict(kinects, try_leave_out_people):
  logger.info('Start run predict')
  for test_kinect in kinects:
    for train_kinect in kinects:
      for test_person in try_leave_out_people:
        percen_sum, percen_max = predict(test_person, test_kinect, train_kinect)
        logger.info('')
        logger.info('Test person: ' + test_person + ' of ' + test_kinect + ' on ' + train_kinect)
        logger.info('Sum pooling correct percentage: ' + str(percen_sum))
        logger.info('Max pooling correct percentage: ' + str(percen_max))

  logger.info('Finished run predict')

if __name__ == '__main__':
  logger.info('Start')
  knn = 5

  kinects = ['Kinect_1', 'Kinect_2', 'Kinect_3', 'Kinect_4', 'Kinect_5']
  try_leave_out_people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuy', 'Tuyen']

  # kinects = ['Kinect_4', 'Kinect_5']

  # Generate vector for each person on each kinect
  # run_gen_vector_data(kinects, try_leave_out_people, knn)

  # Train SVC, save trained SVC and test model
  save_data(kinects, try_leave_out_people)

  # Run test model on trained SVC
  run_predict(kinects, try_leave_out_people)


