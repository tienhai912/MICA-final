import numpy as np
import os
import zipfile
import pickle

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
  check_folder_exist(file_path)
  with open(file_path, 'wb') as f:
    np.savetxt(f, matrix, fmt='%7f', delimiter='\t')


def remove_non_origin_feature_column(feature_matrix, num_col=10):
  return np.delete(feature_matrix, np.s_[:num_col], axis=1)

# (2, 3) + (2, 5) = (2, 8)
def merge_two_matrices_row_by_row(matrix1, matrix2):
  return np.concatenate((matrix1, matrix2), axis=1)

# (2, 3) + (4, 3) = (6, 3)
def merge_two_matrices_col_by_col(matrix1, matrix2):
  return np.concatenate((matrix1, matrix2), axis=0)

def save_model(model, file_path):
  check_folder_exist(file_path)
  with open(file_path, 'wb') as f:
    pickle.dump(model, f)

def load_model(file_path):
  with open(file_path, 'rb') as f:
    model = pickle.load(f)

  return model

def sort_matrix_by_column_desc(matrix, column_num=5):
  return matrix[matrix[:, column_num].argsort()[::-1]]

def check_folder_exist(folder):
  os.makedirs(os.path.dirname(folder), exist_ok=True)


def test():
  B = np.array([[122,2],[4,5],[7,8],[10,11]])
  B = sort_matrix_by_column_desc(B, 0)[:2]
  print(B)

if __name__ == '__main__':
  test()
