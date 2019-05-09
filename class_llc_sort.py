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

class LLC_sort(LLC):
  def __init__(self, k_means=1024, rand_tra_book=200, knn=5, sort_col_num=5, num_tra_data=200,
    input_root_folder='/home/dangmanhtruong95/NTHai/iDT_output/', log_file_name='llc_sort.log'):

    LLC.__init__(self, k_means=k_means, rand_tra_book=rand_tra_book, knn=knn,
    input_root_folder=input_root_folder, log_file_name=log_file_name)

    # sort_col_num = position of attribute use to sort iDT code, default = 5 (trajectory length)
    self.sort_col_num = sort_col_num
    # num_tra_data = number of trajectories taken to get LLC code, default = 200
    self.num_tra_data = num_tra_data

    # output_codebook_root_folder = root folder of trained codebook
    self.output_codebook_root_folder = '/home/dangmanhtruong95/NTHai/iDT_output/'
    # output_root_folder = root folder of llc data and SVM
    self.output_root_folder = '/home/dangmanhtruong95/NTHai/iDT_output_llc_sort_' + str(num_tra_data) + '/'

    # Create output folder if not exist
    support.check_folder_exist(self.output_codebook_root_folder)
    support.check_folder_exist(self.output_root_folder)

  # Get iDT output without first 10 columns (only take trajectories, HOG, HOF, MBH)
  def get_origin_trajectories(self, input_folder):
    support.unzip_file(os.path.join(input_folder, self.output_zip_file), input_folder)

    feature_matrix = support.read_features_out(os.path.join(input_folder, self.out_feature_file))

    if (self.sort_col_num > 0) and (self.num_tra_data > 0):
      feature_matrix = (support.sort_matrix_by_column_desc(feature_matrix, column_num=self.sort_col_num))[:self.num_tra_data]

    feature_matrix = support.remove_non_origin_feature_column(feature_matrix)

    os.remove(os.path.join(input_folder, self.out_feature_file))

    return feature_matrix


if __name__ == '__main__':

  first_llc = LLC_sort(log_file_name='llc_sort.log')
  first_llc.final_run_data_and_train_svm()
