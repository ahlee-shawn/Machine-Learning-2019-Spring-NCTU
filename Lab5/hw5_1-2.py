from libsvm.svmutil import *
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
	return np.array(x_train), np.array(y_train)

def grid_search(x_train, y_train):
	cost = ["1", "2", "3"]
	gamma = ["0.25", "0.5"]
	degree = ["2", "3", "4"]
	coef0 = ["0", "1", "2"]
	best_parameter = ""
	best_accuracy = 0.0
	for i in range(0, 3):
		for j in range(0, 3):
			parameter = "-v 3 -q -t " + str(i) + " -c " + cost[j]
			if i == 0:
				accuracy = svm_train(y_train, x_train, parameter)
				if accuracy > best_accuracy:
					best_accuracy = accuracy
					best_parameter = parameter
			if i == 1:
				for k in range(0, 2):
					for p in range(0, 3):
						for q in range(0, 3):
							new_parameter = parameter + " -g " + gamma[k] + " -d " + degree[p] + " -r " + coef0[q]
							accuracy = svm_train(y_train, x_train, new_parameter)
							if accuracy > best_accuracy:
								best_accuracy = accuracy
								best_parameter = new_parameter
			if i == 2:
				for k in range(0, 2):
					new_parameter = parameter + " -g " + gamma[k]
					accuracy = svm_train(y_train, x_train, new_parameter)
					if accuracy > best_accuracy:
						best_accuracy = accuracy
						best_parameter = new_parameter
	print("Best Accuracy: {}".format(best_accuracy))
	print("Corresponding Parameter: {}".format(best_parameter))

if __name__ == "__main__":
	x_train, y_train = read_csv()
	grid_search(x_train, y_train)