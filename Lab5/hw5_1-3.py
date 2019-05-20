from libsvm.svmutil import *
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np
import csv

def read_csv():
	with open('X_train.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		x_train = list(csv_reader)
		x_train = [[float(y) for y in x] for x in x_train]
	with open('Y_train.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		y_train_2d = list(csv_reader)
		y_train = [y for x in y_train_2d for y in x]
		y_train = [ int(x) for x in y_train ]
	with open('X_test.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		x_test = list(csv_reader)
		x_test = [[float(y) for y in x] for x in x_test]
	with open('Y_test.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		y_test_2d = list(csv_reader)
		y_test = [y for x in y_test_2d for y in x]
		y_test = [ int(x) for x in y_test ]
	return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def compute_kernel(x_train, x_test):
	negative_gamma = -1 / 4 # gamma default: 1/num_features
	train_linear_kernel = np.matmul(x_train, np.transpose(x_train))
	train_rbf_kernel = squareform(np.exp(negative_gamma * pdist(x_train, 'sqeuclidean')))
	x_train_kernel = np.hstack((np.arange(1, 5001).reshape((5000, 1)), np.add(train_linear_kernel, train_rbf_kernel)))

	test_linear_kernel = np.matmul(x_test, np.transpose(x_train))
	test_rbf_kernel = np.exp(negative_gamma * cdist(x_test, x_train, 'sqeuclidean'))
	x_test_kernel = np.hstack((np.arange(1, 2501).reshape((2500, 1)), np.add(test_linear_kernel, test_rbf_kernel)))

	return x_train_kernel, x_test_kernel

def svm(x_train_kernel, y_train, x_test_kernel, y_test):
	prob  = svm_problem(y_train, x_train_kernel, isKernel=True)
	param = svm_parameter('-q -t 4')
	model = svm_train(prob, param)
	svm_predict(y_test, x_test_kernel, model)

if __name__ == "__main__":
	x_train, y_train, x_test, y_test = read_csv()
	x_train_kernel, x_test_kernel = compute_kernel(x_train, x_test)
	svm(x_train_kernel, y_train, x_test_kernel, y_test)