import matplotlib.pyplot as plt
import numpy as np
import csv

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def read_input():
	with open('moon.txt') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		data1 = list(csv_reader)
		data1 = [[float(y) for y in x] for x in data1]
	with open('circle.txt') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		data2 = list(csv_reader)
		data2 = [[float(y) for y in x] for x in data2]
	#return np.append(np.array(data1), np.array(data2)).reshape(3000, 2)
	return np.array(data2).reshape(1500, 2)

def compute_rbf_kernel(data):
	negative_gamma = -0.5
	kernel_data = np.zeros([data.shape[0], data.shape[0]], dtype=np.float64) # kernel_data size: 3000*3000
	for i in range(0, data.shape[0]):
		for j in range(i + 1, data.shape[0]):
			temp = np.exp(negative_gamma * np.linalg.norm(data[i] - data[j]))
			if temp < 0:
				temp = 0
			kernel_data[i][j] = temp
			kernel_data[j][i] = temp
	return kernel_data

def initialization(k, data):
	temp = np.random.randint(low=0, high=1500, size=k)
	means = np.zeros([k, 2], dtype=np.float32)
	for i in range(0, k):
		means[i] = data[temp[i]]
	previous_classification = []
	for i in range(1500):
		if i % 2 == 0:
			previous_classification.append(0)
		else:
			previous_classification.append(1)
	#previous_classification = np.ones([1500], np.int)
	return means, np.asarray(previous_classification), 1 # 1 for iteration

def classify(data, means):
	classification = np.zeros([1500], dtype=int)
	for i in range(0, 1500):
		temp = np.zeros([means.shape[0]], dtype=np.float32) # temp size: k
		for j in range(0, means.shape[0]):
			temp[j] = np.linalg.norm(data[i] - means[j])
			#print(temp[j])
			#print(data[i])
			#print(means[j])
		classification[i] = np.argmin(temp)
	return classification

def calculate_error(classification, previous_classification):
	error = 0
	for i in range(0, 1500):
		error += np.absolute(classification[i] - previous_classification[i])
	return error

def update(data, means, classification):
	means = np.zeros(means.shape, dtype=np.float32)
	count = np.zeros(means.shape, dtype=np.int)
	one = np.ones(means.shape[1], dtype=np.int)
	for i in range(0, 1500):
		means[classification[i]] += data[i]
		count[classification[i]] += one
	for i in range(0, means.shape[0]):
		if count[i][0] == 0:
			count[i] += one
	return np.true_divide(means, count)

def draw(k, data, means, classification):
	color = iter(plt.cm.rainbow(np.linspace(0, 1, means.shape[0] * 2)))
	title = "Spectral-Clustering"
	print(classification)
	plt.title(title)
	plt.clf()
	for i in range(0, means.shape[0]):
		col = next(color)
		for j in range(0, data.shape[0]):
			if classification[j] == i:
				plt.scatter(data[j][0], data[j][1], s=8, c=col)
	'''
	for i in range(0, means.shape[0]):
		col = next(color)
		plt.scatter(means[i][0], means[i][1], s=32, c=col)
	'''
	plt.show()

def k_means(raw_data, data):
	# k is the number of cluster
	k = 2
	means, previous_classification, iteration = initialization(k, data) # means size: k*2 previous_classification: 3000
	classification = classify(data, means) # classification: 3000
	error = calculate_error(classification, previous_classification)
	print(classification)
	#draw(k, data, means, classification)
	while(True):
		iteration += 1
		means = update(data, means, classification)
		previous_classification = classification
		classification = classify(data, means)
		error = calculate_error(classification, previous_classification)
		print(classification)
		print(error)
		if error < 5:
			break
	#draw(k, data, means, classification)
	print(means)
	draw(k, raw_data, means, classification)

if __name__ == "__main__":
	k = 2
	data = read_input()
	Weight = compute_rbf_kernel(data) # Weight size: 3000 * 3000
	Degree = np.diag(Weight.sum(axis=1)) 
	print(Degree)
	Laplacian = Degree - Weight
	eigen_values, eigen_vectors = np.linalg.eig(Laplacian)
	idx = np.argsort(eigen_values)
	eigen_vectors = eigen_vectors[idx][::-1]
	U = (eigen_vectors[:,:k])[:,1:]
	#draw_high(U)
	print(U.shape)
	k_means(data, U)