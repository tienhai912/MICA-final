import numpy as np
import os
import scipy.cluster.vq as vq
import zipfile
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import code_support as support
from datetime import datetime
import logging

class LLC:
  def __init__(self, k_means=1024, rand_tra_book=200, knn=5,
    input_root_folder='/home/dangmanhtruong95/NTHai/iDT_output/', log_file_name='llc.log'):

    # For training codebook:
    # k_means = number of codebook vectors, default = 1024
    self.k_means = k_means
    # ran_tra_book = number of vectors take from each video
    self.rand_tra_book = rand_tra_book

    # For LLC
    # knn = number of nearest neighbors, use for LLC, default = 5
    self.knn = knn

    # input_root_folder = iDT folder
    self.input_root_folder = input_root_folder
    self.output_codebook_root_folder = '/home/dangmanhtruong95/NTHai/iDT_output/'
    self.output_root_folder = '/home/dangmanhtruong95/NTHai/iDT_output_llc/'

    # Create output folder if not exist
    support.check_folder_exist(self.output_codebook_root_folder)
    support.check_folder_exist(self.output_root_folder)

    # List of input folders
    self.kinects = ['Kinect_1', 'Kinect_2', 'Kinect_3', 'Kinect_4', 'Kinect_5']
    self.people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuy', 'Tuyen']
    self.try_leave_out_people = ['Giang', 'Hai', 'Long', 'Minh', 'Thuy', 'Tuyen']

    # List of file names
    self.code_vector_folder_name = 'code_vector'
    self.svm_test_data_folder_name = 'svm_test_data'
    self.train_svm_folder_name = 'trained_svm'
    self.csv_folder_name = 'csv'

    self.out_feature_file = 'out_features.txt'
    self.output_zip_file = 'output.zip'

    # Log
    for handler in logging.root.handlers[:]:
      logging.root.removeHandler(handler)

    logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    self.logger = logging.getLogger('Hai')


  def get_random_origin_trajectories(self, input_folder):
    support.unzip_file(os.path.join(input_folder, self.output_zip_file), input_folder)

    feature_matrix = support.read_features_out(os.path.join(input_folder, self.out_feature_file))
    self.logger.info(os.path.join(input_folder, self.out_feature_file))
    self.logger.info(feature_matrix.shape)

    if (feature_matrix.shape[0] >= self.rand_tra_book):
      feature_matrix = feature_matrix[np.random.choice(feature_matrix.shape[0], self.rand_tra_book, replace=False)]
    else:
      feature_matrix = feature_matrix[np.random.choice(feature_matrix.shape[0], self.rand_tra_book, replace=True)]

    feature_matrix = support.remove_non_origin_feature_column(feature_matrix, num_col=10)

    os.remove(os.path.join(input_folder, self.out_feature_file))

    return feature_matrix

  def run_get_origin_trajectories(self, kinects, try_leave_out_people):
    out_vector = {}
    for kinect in self.kinects:
      out_vector[kinect] = {}
      for test_person in self.try_leave_out_people:
        out_vector[kinect][test_person] = np.empty((0, 426))
        train_people = [person for person in self.people if person != test_person]
        for person in train_people:
          folder_list = support.list_all_folders_in_a_directory(os.path.join(self.input_root_folder, person, kinect))

          for folder in folder_list:
            input_features_folder = os.path.join(self.input_root_folder, person, kinect, folder)
            feature_vector = self.get_random_origin_trajectories(input_features_folder)

            out_vector[kinect][test_person] = support.merge_two_matrices_col_by_col(out_vector[kinect][test_person], feature_vector)

        self.logger.info('Finished get Trajectories to test ' + test_person + ' in ' + kinect)

      self.logger.info('Finished get Trajectories in ' + kinect)

    return out_vector

  def gen_origin_codebook(self):
    support.check_folder_exist(os.path.join(self.output_codebook_root_folder, 'codebook'))
    out_vector = self.run_get_origin_trajectories(self.kinects, self.try_leave_out_people)
    self.logger.info('Finished get all Trajectories')

    for kinect in self.kinects:
      for test_person in self.try_leave_out_people:
        temp_out_vector = out_vector[kinect][test_person]
        self.logger.info('Input out_vector for ' + kinect + '_' + test_person + '_codebook.txt with shape ' + str(temp_out_vector.shape))

        train_k_means = KMeans(init='k-means++', n_clusters=self.k_means, n_init=10)
        train_k_means.fit(temp_out_vector)
        codebook = train_k_means.cluster_centers_

        self.logger.info('Finished gen ' + kinect + '_' + test_person + '_codebook.txt with shape ' + str(codebook.shape))
        codebook_file = os.path.join(self.output_codebook_root_folder, 'codebook', kinect + '_' + test_person + '_codebook.txt')

        with open(codebook_file,'wb') as f:
          np.savetxt(f, codebook, fmt='%7f', delimiter='\t')

        self.logger.info('Finished save ' + kinect + '_' + test_person + '_codebook.txt')

  # SVM
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

  # Get iDT output without first 10 columns (only take trajectories, HOG, HOF, MBH)
  def get_origin_trajectories(self, input_folder):
    support.unzip_file(os.path.join(input_folder, self.output_zip_file), input_folder)

    feature_matrix = support.read_features_out(os.path.join(input_folder, self.out_feature_file))

    feature_matrix = support.remove_non_origin_feature_column(feature_matrix)

    os.remove(os.path.join(input_folder, self.out_feature_file))

    return feature_matrix

  # Vectorize iDT code and save result vectors with their corresponding label as txt
  def gen_vector_data(self, kinect, codebook_kinect, test_person):

    codebook_file = os.path.join(self.output_codebook_root_folder, 'codebook', codebook_kinect + '_' + test_person + '_codebook.txt')
    B = support.read_features_out(codebook_file)

    save_vector_folder = os.path.join(self.output_root_folder, self.code_vector_folder_name)

    for person in self.people:
      temp_X_sum = np.empty((0, 1024))
      temp_X_max = np.empty((0, 1024))
      temp_y = np.empty((0, 1))
      folder_list = support.list_all_folders_in_a_directory(os.path.join(self.input_root_folder, person, kinect))

      for folder in folder_list:
        input_features_folder = os.path.join(self.input_root_folder, person, kinect, folder)
        feature_vector = self.get_origin_trajectories(input_features_folder)

        code_vector = self.calculate_llc(B, feature_vector)
        # Sum pooling and max pooling
        code_vector_sum = np.sum(code_vector, axis=0, keepdims=True)
        code_vector_max = np.amax(code_vector, axis=0, keepdims=True)
        # Norm 2?
        code_vector_sum = code_vector_sum/np.sqrt(np.sum(np.square(code_vector_sum)))
        code_vector_max = code_vector_max/np.sqrt(np.sum(np.square(code_vector_max)))
        # label
        label_vector = np.array([[float(folder.split('_')[0])]])

        temp_X_sum = support.merge_two_matrices_col_by_col(temp_X_sum, code_vector_sum)
        temp_X_max = support.merge_two_matrices_col_by_col(temp_X_max, code_vector_max)
        temp_y = support.merge_two_matrices_col_by_col(temp_y, label_vector)


      # kinect = kinect of input data
      # codebook_kinect = kinect of codebook
      prefix_vector_name = kinect + '_to_' + codebook_kinect + '_' + person + '_' + test_person +'_'

      self.logger.info(prefix_vector_name)
      # self.logger.info('temp_X_sum: ' + str(temp_X_sum.shape))
      # self.logger.info('temp_X_max: ' + str(temp_X_max.shape))
      # self.logger.info('temp_y: ' + str(temp_y.shape))

      support.save_features_out(temp_X_sum, os.path.join(save_vector_folder, prefix_vector_name + 'X_sum.txt'))
      support.save_features_out(temp_X_max, os.path.join(save_vector_folder, prefix_vector_name + 'X_max.txt'))
      support.save_features_out(temp_y, os.path.join(save_vector_folder, prefix_vector_name + 'y.txt'))

  # Run gen_vector_data on a subset of the input data
  def run_gen_vector_data(self):
    self.logger.info('Start run gen vector data')

    for kinect in self.kinects:
      for test_person in self.try_leave_out_people:
        self.gen_vector_data(kinect, kinect, test_person)

    self.logger.info('Finished run gen vector data')

  # Read vector files (output of run_gen_vector_data)
  def read_vector_data(self, kinect, codebook_kinect, person, test_person):
    save_vector_folder = os.path.join(self.output_root_folder, code_vector_folder_name)
    prefix_vector_name = kinect + '_to_' + codebook_kinect + '_' + person + '_' + test_person +'_'

    temp_X_sum = support.read_features_out(os.path.join(save_vector_folder, prefix_vector_name + 'X_sum.txt'))
    temp_X_max = support.read_features_out(os.path.join(save_vector_folder, prefix_vector_name + 'X_max.txt'))
    temp_y = support.read_features_out(os.path.join(save_vector_folder, prefix_vector_name + 'y.txt'))

    return temp_X_sum, temp_X_max, temp_y

  # Divide vector files into train and test sets
  # Train through a SVC
  # Save predictions and test label
  def save_data(self, kinects, try_leave_out_people):
    self.logger.info('Start save data')
    svm_test_data_folder = os.path.join(self.output_root_folder, self.svm_test_data_folder_name)

    for kinect in self.kinects:
      for test_person in self.try_leave_out_people:
        data = {}
        data['X_train_sum'] = np.empty((0, 1024))
        data['X_test_sum'] = np.empty((0, 1024))
        data['X_train_max'] = np.empty((0, 1024))
        data['X_test_max'] = np.empty((0, 1024))
        data['y_train'] = np.empty((0, 1))
        data['y_test'] = np.empty((0, 1))

        for person in self.people:
          temp_X_sum, temp_X_max, temp_y = self.read_vector_data(kinect, kinect, person, test_person)
          if (person != test_person):
            data['X_train_sum'] = support.merge_two_matrices_col_by_col(data['X_train_sum'], temp_X_sum)
            data['X_train_max'] = support.merge_two_matrices_col_by_col(data['X_train_max'], temp_X_max)
            data['y_train'] = support.merge_two_matrices_col_by_col(data['y_train'], temp_y)
          else:
            data['X_test_sum'] = support.merge_two_matrices_col_by_col(data['X_test_sum'], temp_X_sum)
            data['X_test_max'] = support.merge_two_matrices_col_by_col(data['X_test_max'], temp_X_max)
            data['y_test'] = support.merge_two_matrices_col_by_col(data['y_test'], temp_y)

        # Create and train
        sum_svclassifier = SVC(kernel='linear')
        max_svclassifier = SVC(kernel='linear')
        sum_svclassifier.fit(data['X_train_sum'], data['y_train'].ravel())
        max_svclassifier.fit(data['X_train_max'], data['y_train'].ravel())

        # kinect = kinect train
        # test_person = leave one out person
        svc_prefix_name = kinect + '_' + test_person + '_'
        support.save_model(sum_svclassifier, os.path.join(self.output_root_folder, self.train_svm_folder_name, svc_prefix_name + 'sum.pkl'))
        support.save_model(max_svclassifier, os.path.join(self.output_root_folder, self.train_svm_folder_name, svc_prefix_name + 'max.pkl'))

        # folder to save data of each leave one out test
        # fisrt kinect = kinect of input data
        # second kinect = kinect of codebook
        # test_person = the person got left out to test (the remaining people are used to train)
        save_test_folder = os.path.join(svm_test_data_folder, kinect + '_to_' + kinect + '_' + test_person)

        self.logger.info(save_test_folder)
        # self.logger.info('X_test_sum: ' + str(data['X_test_sum'].shape))
        # self.logger.info('X_test_max: ' + str(data['X_test_max'].shape))
        # self.logger.info('y_test: ' + str(data['y_test'].shape))

        support.save_features_out(data['X_test_sum'], os.path.join(save_test_folder, 'X_test_sum.txt'))
        support.save_features_out(data['X_test_max'], os.path.join(save_test_folder, 'X_test_max.txt'))
        support.save_features_out(data['y_test'], os.path.join(save_test_folder, 'y_test.txt'))

    self.logger.info('Finished save data')

  def predict(self, test_person, test_kinect, train_kinect):
    sum_svc = support.load_model(os.path.join(self.output_root_folder, self.train_svm_folder_name, train_kinect + '_' + test_person + '_' + 'sum.pkl'))
    max_svc = support.load_model(os.path.join(self.output_root_folder, self.train_svm_folder_name, train_kinect + '_' + test_person + '_' + 'max.pkl'))

    test_data_folder = os.path.join(self.output_root_folder, self.svm_test_data_folder_name, test_kinect + '_to_' + test_kinect + '_' + test_person)
    X_test_sum = read_features_out(os.path.join(test_data_folder, 'X_test_sum.txt'))
    X_test_max = read_features_out(os.path.join(test_data_folder, 'X_test_max.txt'))
    y_test = read_features_out(os.path.join(test_data_folder, 'y_test.txt'))

    y_pred_sum = sum_svc.predict(X_test_sum)
    y_pred_max = max_svc.predict(X_test_max)

    percen_sum = np.sum(1 for i, j in zip(y_pred_sum, y_test) if i == j)/len(y_test)
    percen_max = np.sum(1 for i, j in zip(y_pred_max, y_test) if i == j)/len(y_test)

    return percen_sum, percen_max

  def run_predict(self):
    self.logger.info('Start run predict')
    sum_result= {}
    max_result= {}
    num_of_tries = len(try_leave_out_people)

    for test_kinect in self.kinects:
      sum_result[test_kinect] = {}
      max_result[test_kinect] = {}
      for train_kinect in self.kinects:
        temp_sum = 0
        temp_max = 0

        for test_person in self.try_leave_out_people:
          percen_sum, percen_max = self.predict(test_person, test_kinect, train_kinect)
          temp_sum = temp_sum + percen_sum
          temp_max = temp_max + percen_max
          self.logger.info('')
          self.logger.info('Test person: ' + test_person + ' of ' + test_kinect + ' on ' + train_kinect)
          self.logger.info('Sum pooling correct percentage: ' + str(percen_sum))
          self.logger.info('Max pooling correct percentage: ' + str(percen_max))


        sum_result[test_kinect][train_kinect] = temp_sum / num_of_tries
        max_result[test_kinect][train_kinect] = temp_max / num_of_tries


    csv_folder = os.path.join(self.output_root_folder, self.csv_folder_name)
    support.check_folder_exist(os.path.join(csv_folder, 'sum_pooling_result.csv'))
    df_sum = pd.DataFrame(sum_result)
    df_sum.to_csv(os.path.join(csv_folder, 'sum_pooling_result.csv'))
    df_max = pd.DataFrame(max_result)
    df_max.to_csv(os.path.join(csv_folder, 'max_pooling_result.csv'))

    self.logger.info('Finished run predict')


  def final_run_all(self):
    self.logger.info('Start All')

    self.final_run_codebook()
    self.final_run_data()
    self.final_run_train_svm()

    self.looger.info('Finished All')

  def final_run_codebook(self):
    self.gen_origin_codebook()

  def final_run_data(self):
    # Generate vector for each person on each kinect
    self.run_gen_vector_data()

  def final_run_train_svm(self):
    # Train SVC, save trained SVC and test model
    self.save_data()

    # Run test model on trained SVC
    self.run_predict()

  def final_run_data_and_train_svm(self):
    self.final_run_data()
    self.final_run_train_svm()


if __name__ == '__main__':

  first_llc = LLC()
  first_llc.final_run_all()
