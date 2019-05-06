import numpy as np
import os
import scipy.cluster.vq as vq
import zipfile
from datetime import datetime
import logging
from sklearn.cluster import KMeans

for handler in logging.root.handlers[:]:
  logging.root.removeHandler(handler)

logging.basicConfig(filename='codebook_20190427.log', level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('Hai')

out_feature_file = 'out_features.txt'
output_zip_file = 'output.zip'

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

def remove_non_feature_column(feature_matrix):
  return np.delete(feature_matrix, np.s_[:11], axis=1)

# (2, 3) + (2, 5) = (2, 8)
def merge_two_matrices_row_by_row(matrix1, matrix2):
  return np.concatenate((matrix1, matrix2), axis=1)

# (2, 3) + (4, 3) = (6, 3)
def merge_two_matrices_col_by_col(matrix1, matrix2):
  return np.concatenate((matrix1, matrix2), axis=0)

# -----Select 200 longest features
def get_train_vector(input_folder):
  sorted_by_length = 'out_features_sorted_by_length.txt'
  sorted_by_length_zip = 'out_features_sorted_by_length.zip'

  unzip_file(os.path.join(input_folder, sorted_by_length_zip), input_folder)

  feature_matrix = read_features_out(os.path.join(input_folder, sorted_by_length))

  feature_matrix = remove_non_feature_column(feature_matrix)

  os.remove(os.path.join(input_folder, sorted_by_length))

  return feature_matrix.T

def run_get_train_vector():
  input_root_folder = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\'
  # input_root_folder = '/home/dangmanhtruong95/NTHai/iDT_output/'

  # people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuan', 'Thuy', 'Tuyen']
  # people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuan', 'Thuy']
  # kinects = ['Kinect_1', 'Kinect_2', 'Kinect_3', 'Kinect_4', 'Kinect_5']

  people = ['Giang']
  kinects = ['Kinect_1', 'Kinect_2']

  out_vector = np.empty((0, 426))

  for person in people:
    for kinect in kinects:
      folder_list = list_all_folders_in_a_directory(os.path.join(input_root_folder, person, kinect))
      # folder_list = ['10_1']
      for folder in folder_list:

        input_features_folder = os.path.join(input_root_folder, person, kinect, folder)
        feature_vector = get_train_vector(input_features_folder)

        out_vector = merge_two_matrices_col_by_col(out_vector, feature_vector.T)

  return out_vector

def gen_codebook(k_means, codebook_file):

  out_vector = run_get_train_vector()
  logger.info('Finished get train vector')

  out_vector = vq.whiten(out_vector)
  codebook, distortion = vq.kmeans2(out_vector, k=k_means, minit='++')

  # with open('C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\output\\test_codebook.txt','wb') as f:
  #   np.savetxt(f, codebook, fmt='%7f', delimiter='\t')
  logger.info('Finished gen codebook')

  with open(codebook_file,'wb') as f:
    np.savetxt(f, codebook, fmt='%7f', delimiter='\t')

# -----Randomly select 200 features
def remove_non_origin_feature_column(feature_matrix):
  return np.delete(feature_matrix, np.s_[:10], axis=1)

def get_200_origin_trajectories(input_folder):
  unzip_file(os.path.join(input_folder, output_zip_file), input_folder)

  feature_matrix = read_features_out(os.path.join(input_folder, out_feature_file))
  logger.info(os.path.join(input_folder, out_feature_file))
  logger.info(feature_matrix.shape)

  if (feature_matrix.shape[0] >= 200):
    feature_matrix = feature_matrix[np.random.choice(feature_matrix.shape[0], 200, replace=False)]
  else:
    feature_matrix = feature_matrix[np.random.choice(feature_matrix.shape[0], 200, replace=True)]

  feature_matrix = remove_non_origin_feature_column(feature_matrix)

  os.remove(os.path.join(input_folder, out_feature_file))

  return feature_matrix

def run_get_origin_trajectories(kinects, try_leave_out_people):
  # input_root_folder = 'C:\\Users\\TienHai\\Desktop\\iDT\\run_LLC\\'
  input_root_folder = '/home/dangmanhtruong95/NTHai/iDT_output/'

  # people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuan', 'Thuy', 'Tuyen']
  people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuy', 'Tuyen']

  out_vector = {}

  for kinect in kinects:
    out_vector[kinect] = {}
    for test_person in try_leave_out_people:
      out_vector[kinect][test_person] = np.empty((0, 426))
      train_people = [person for person in people if person != test_person]
      for person in train_people:
        folder_list = list_all_folders_in_a_directory(os.path.join(input_root_folder, person, kinect))
        # folder_list = ['10_1']
        for folder in folder_list:
          input_features_folder = os.path.join(input_root_folder, person, kinect, folder)
          feature_vector = get_200_origin_trajectories(input_features_folder)
          # print(feature_vector.shape)
          # print(out_vector[kinect][test_person].shape)

          out_vector[kinect][test_person] = merge_two_matrices_col_by_col(out_vector[kinect][test_person], feature_vector)

      logger.info('Finished get Trajectories to test ' + test_person + ' in ' + kinect)

    logger.info('Finished get Trajectories in ' + kinect)

  return out_vector

def gen_origin_codebook(kinects, try_leave_out_people, k_means):
  input_root_folder = '/home/dangmanhtruong95/NTHai/iDT_output/'

  people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuy', 'Tuyen']

  out_vector = run_get_origin_trajectories(kinects, try_leave_out_people)
  logger.info('Finished get all Trajectories')

  for kinect in kinects:
    for test_person in try_leave_out_people:
      # temp_out_vector = vq.whiten(out_vector[kinect][test_person])
      temp_out_vector = out_vector[kinect][test_person]
      logger.info('Input out_vector for ' + kinect + '_' + test_person + '_codebook.txt with shape ' + str(temp_out_vector.shape))
      # codebook, distortion = vq.kmeans2(temp_out_vector, k=k_means, minit='++')
      train_k_means = KMeans(init='k-means++', n_clusters=k_means, n_init=10)
      train_k_means.fit(temp_out_vector)
      codebook = train_k_means.cluster_centers_

      logger.info('Finished gen ' + kinect + '_' + test_person + '_codebook.txt with shape ' + str(codebook.shape))
      codebook_file = os.path.join(input_root_folder, 'codebook', kinect + '_' + test_person + '_codebook.txt')

      with open(codebook_file,'wb') as f:
        np.savetxt(f, codebook, fmt='%7f', delimiter='\t')

      logger.info('Finished save ' + kinect + '_' + test_person + '_codebook.txt')

if __name__ == '__main__':
  logger.info('Start')

  kinects = ['Kinect_1', 'Kinect_2', 'Kinect_3', 'Kinect_4', 'Kinect_5']
  try_leave_out_people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuy', 'Tuyen']
  # kinects = ['Kinect_4', 'Kinect_5']

  k_means = 1024
  # codebook_file = os.path.join(input_root_folder, 'codebook', '5_people_codebook.txt')
  # gen_codebook(k_means, codebook_file)
  gen_origin_codebook(kinects, try_leave_out_people, k_means)

  logger.info('Finished')
