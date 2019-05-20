import matplotlib.pyplot as plt
from libsvm.svmutil import *
from scipy.spatial.distance import pdist, squareform
import numpy as np
import csv

def read_csv():
	with open('Plot_X.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		x_data = list(csv_reader)
		x_data = [[float(y) for y in x] for x in x_data]
	with open('Plot_Y.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		y_train_2d = list(csv_reader)
		y_data = [y for x in y_train_2d for y in x]
		y_data = [ int(x) for x in y_data ]
	return np.array(x_data), np.array(y_data)

def compute_kernel(x):
	negative_gamma = -1 / 4 # gamma default: 1/num_features
	linear_kernel = np.matmul(x, np.transpose(x))
	rbf_kernel = squareform(np.exp(negative_gamma * pdist(x, 'sqeuclidean')))
	x_kernel = np.hstack((np.arange(1, 3001).reshape((3000, 1)), np.add(linear_kernel, rbf_kernel)))
	return x_kernel

def split_cluster(x, p_labels):
	cluster1 = []
	cluster2 = []
	cluster3 = []
	for j in range(0, 3000):
		if p_labels[j] == 0.0:
			cluster1.append(x[j])
		if p_labels[j] == 1.0:
			cluster2.append(x[j])
		if p_labels[j] == 2.0:
			cluster3.append(x[j])
	return cluster1, cluster2, cluster3

def draw(x, y, cluster1, cluster2, cluster3, title, graph_number, model):
	support_vector_x_1 = []
	support_vector_y_1 = []
	support_vector_x_2 = []
	support_vector_y_2 = []
	support_vector_x_3 = []
	support_vector_y_3 = []
	for i in range(0, model.l):
		if y[model.sv_indices[i]-1] == 0.0:
			support_vector_x_1.append(x[model.sv_indices[i]-1][0])
			support_vector_y_1.append(x[model.sv_indices[i]-1][1])
		if y[model.sv_indices[i]-1] == 1.0:
			support_vector_x_2.append(x[model.sv_indices[i]-1][0])
			support_vector_y_2.append(x[model.sv_indices[i]-1][1])
		if y[model.sv_indices[i]-1] == 2.0:
			support_vector_x_3.append(x[model.sv_indices[i]-1][0])
			support_vector_y_3.append(x[model.sv_indices[i]-1][1])
	plt.subplot(graph_number)
	plt.title(title)
	plt.scatter(cluster1[:,0], cluster1[:,1], color = 'black', s = 8)
	plt.scatter(cluster2[:,0], cluster2[:,1], color = 'green', s = 8)
	plt.scatter(cluster3[:,0], cluster3[:,1], color = 'blue', s = 8)
	plt.scatter(support_vector_x_1, support_vector_y_1, color = 'red', marker = 'x', s = 4)
	plt.scatter(support_vector_x_2, support_vector_y_2, color = 'yellow', marker = 's', s = 4)
	plt.scatter(support_vector_x_3, support_vector_y_3, color = 'pink', marker = '^', s = 4)
	plt.draw()

def svm(x, y):
	kernel = ["linear", "polynomial", "RBF", "linear + RBF"]
	graph_number = [221, 222, 223, 224]
	for i in range(0, 4):
		if i != 3:
			parameter = "-q -t " + str(i)
			model = svm_train(y, x, parameter)
			p_labels, p_acc, p_vals = svm_predict(y, x, model)
		else:
			x_kernel = compute_kernel(x)			
			prob = svm_problem(y, x_kernel, isKernel=True)
			param = svm_parameter('-q -t 4')
			model = svm_train(prob, param)
			p_labels, p_acc, p_vals = svm_predict(y, x_kernel, model)
		cluster1, cluster2, cluster3 = split_cluster(x, p_labels)
		draw(x, y, np.array(cluster1), np.array(cluster2), np.array(cluster3), kernel[i], graph_number[i], model)

if __name__ == "__main__":
	x, y= read_csv()
	svm(x, y)
	plt.tight_layout()
	plt.show()