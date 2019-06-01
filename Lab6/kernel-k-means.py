import matplotlib.pyplot as plt
import numpy as np
import csv

from scipy.spatial.distance import cdist, pdist, squareform

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def read_input(dataset):
	if dataset == "test.txt":
		with open(dataset) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=' ')
			data1 = list(csv_reader)
			data1 = [[float(y) for y in x] for x in data1]
		return np.array(data1).reshape(400, 2)
	else:
		with open(dataset) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			data1 = list(csv_reader)
			data1 = [[float(y) for y in x] for x in data1]
		return np.array(data1).reshape(1500, 2)

def compute_rbf_kernel(data, dataset):
	if dataset == "test.txt":
		sigma = 5
	else:
		sigma = 0.1
	kernel_data = np.zeros([data.shape[0], data.shape[0]], dtype=np.float32) # kernel_data size: 3000*3000
	for i in range(0, data.shape[0]):
		for j in range(i + 1, data.shape[0]):
			delta = abs(np.subtract(data[i,:], data[j,:]))
			squaredEuclidean = np.square(delta).sum(axis=0)
			temp = np.exp(-(squaredEuclidean) / (2 * sigma ** 2))
			kernel_data[i][j] = temp
			kernel_data[j][i] = temp
	
	return kernel_data

def initialization(data, k):
	means = np.random.rand(k, 2)
	previous_classification = []
	for i in range(data.shape[0]):
		if i % 2 == 1:
			previous_classification.append(0)
		else:
			previous_classification.append(1)
	return means, np.asarray(previous_classification), 1, 0 # 1 for iteration 0 for prev_error

def second_term_of_calculate_distance(data, kernel_data, classification, data_number, cluster_number, k):
	result = 0
	number_in_cluster = 0
	for i in range(0, data.shape[0]):
		if classification[i] == cluster_number:
			number_in_cluster += 1
	if number_in_cluster == 0:
		number_in_cluster = 1
	for i in range(0, data.shape[0]):
		if classification[i] == cluster_number:
			result += kernel_data[data_number][i]
	return -2 * (result / number_in_cluster)

def third_term_of_calculate_distance(kernel_data, classification, k):
	temp = np.zeros(k, dtype=np.float32)
	temp1 = np.zeros(k, dtype=np.float32)
	for i in range(0, classification.shape[0]):
		temp[classification[i]] += 1
	for i in range(0, k):
		for p in range(0, kernel_data.shape[0]):
			for q in range(p + 1, kernel_data.shape[1]):
				if classification[p] == i and classification[q] == i:
					temp1[i] += kernel_data[p][q]
	for i in range(0, k):
		if temp[i] == 0:
			temp[i] = 1
		temp1[i] /= (temp[i] ** 2)
	return temp1

def classify(data, kernel_data, means, classification):
	temp_classification = np.zeros([data.shape[0]], dtype=np.int)
	third_term = third_term_of_calculate_distance(kernel_data, classification, means.shape[0])
	for i in range(0, data.shape[0]):
		temp = np.zeros([means.shape[0]], dtype=np.float32) # temp size: k
		for j in range(0, means.shape[0]):
			temp[j] = second_term_of_calculate_distance(data, kernel_data, classification, i, j, means.shape[0]) + third_term[j]
		temp_classification[i] = np.argmin(temp)
	return temp_classification

def calculate_error(classification, previous_classification):
	error = 0
	for i in range(0, classification.shape[0]):
		error += np.absolute(classification[i] - previous_classification[i])
	return error

def update(data, means, classification):
	means = np.zeros(means.shape, dtype=np.float32)
	count = np.zeros(means.shape, dtype=np.int)
	one = np.ones(means.shape[1], dtype=np.int)
	for i in range(0, classification.shape[0]):
		means[classification[i]] += data[i]
		count[classification[i]] += one
	return np.true_divide(means, count)

def draw(k, data, means, classification, iteration, dataset):
	color = iter(plt.cm.rainbow(np.linspace(0, 1, k * 2)))
	title = "Kernel-K-Means Iteration-" + str(iteration)
	plt.clf()
	for i in range(0, k):
		col = next(color)
		for j in range(0, data.shape[0]):
			if classification[j] == i:
				plt.scatter(data[j][0], data[j][1], s=8, c=col)
	for i in range(0, k):
		col = next(color)
		plt.scatter(means[i][0], means[i][1], s=32, c=col)
	plt.suptitle(title)
	if dataset == "moon.txt":
		plt.savefig("./Screenshots/Kernel-K-Means/moon/" + title + ".png")
	elif dataset == "circle.txt":
		plt.savefig("./Screenshots/Kernel-K-Means/circle/" + title + ".png")
	else:
		plt.savefig("./Screenshots/Kernel-K-Means/test/" + title + ".png")
	#plt.savefig(title+'.png')


def kernel_k_means(data, kernel_data, dataset):
	# k is the number of cluster
	k = 2
	means, previous_classification, iteration, prev_error = initialization(data, k) # means size: k*2 previous_classification: 3000
	draw(k, data, means, previous_classification, iteration, dataset)
	classification = classify(data, kernel_data, means, previous_classification) # classification: 3000
	error = calculate_error(classification, previous_classification)
	while(True):
		draw(k, data, means, classification, iteration, dataset)
		iteration += 1
		previous_classification = classification
		classification = classify(data, kernel_data, means, classification)
		error = calculate_error(classification, previous_classification)
		print(error)
		if error == prev_error:
			break
		prev_error = error
	means = update(data, means, classification)
	draw(k, data, means, classification, iteration, dataset)

if __name__ == "__main__":
	dataset = "moon.txt"
	data = read_input(dataset) # data size: 3000*2
	kernel_data = compute_rbf_kernel(data, dataset)
	kernel_k_means(data, kernel_data, dataset)
	