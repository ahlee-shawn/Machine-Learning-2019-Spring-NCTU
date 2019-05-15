from libsvm.svmutil import *
import numpy as np
import csv

def read_csv():
	x_train = []
	y_train = []
	x_test = []
	y_test = []
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
	return x_train, y_train, x_test, y_test

def svm(x_train, y_train, x_test, y_test):
	kernel = ["linear", "polynomial", "RBF"]
	for i in range(0, 3):
		print("Kernel Function: {}".format(kernel[i]))
		parameter = "-q -t " + str(i)
		model = svm_train(y_train, x_train, parameter)
		p_labels, p_acc, p_vals = svm_predict(y_test, x_test, model)

if __name__ == "__main__":
	x_train, y_train, x_test, y_test = read_csv()
	svm(x_train, y_train, x_test, y_test)