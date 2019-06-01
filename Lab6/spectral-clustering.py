import matplotlib.pyplot as plt
import numpy as np
import csv

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def read_input(dataset):
	with open(dataset) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		data1 = list(csv_reader)
		data1 = [[float(y) for y in x] for x in data1]
	return np.array(data1).reshape(1500, 2)

def compute_rbf_kernel(data):
	sigma = 0.1
	kernel_data = np.zeros([data.shape[0], data.shape[0]], dtype=np.float64) # kernel_data size: 3000*3000
	for i in range(0, data.shape[0]):
		for j in range(i + 1, data.shape[0]):
			delta = abs(np.subtract(data[i,:], data[j,:]))
			squaredEuclidean = np.square(delta).sum(axis=0)
			temp = np.exp(-(squaredEuclidean) / (2 * sigma ** 2))
			# If temp < 0, the similarity is 0.
			# however, since it's a RBF kernel, temp < 0 is never true
			kernel_data[i][j] = temp
			kernel_data[j][i] = temp
	return kernel_data

def initialization(k, data):
	temp = np.random.randint(low=0, high=data.shape[0], size=k)
	means = np.zeros([k, k], dtype=np.float32)
	for i in range(0, k):
		means[i,:] = data[temp[i],:]
	previous_classification = []
	for i in range(data.shape[0]):
		if i % 2 == 0:
			previous_classification.append(0)
		else:
			previous_classification.append(1)
	return means, np.asarray(previous_classification), 1 # 1 for iteration

def classify(data, means):
	classification = np.zeros([data.shape[0]], dtype=int)
	for i in range(0, data.shape[0]):
		temp = np.zeros([means.shape[0]], dtype=np.float32) # temp size: k
		for j in range(0, means.shape[0]):
			delta = abs(np.subtract(data[i,:], means[j,:]))
			temp[j] = np.square(delta).sum(axis=0)
		classification[i] = np.argmin(temp)
	return classification

def calculate_error(classification, previous_classification):
	error = 0
	for i in range(0, classification.shape[0]):
		error += np.absolute(classification[i] - previous_classification[i])
	return error

def update(data, means, classification):
	means = np.zeros(means.shape, dtype=np.float32)
	count = np.zeros(means.shape, dtype=np.int)
	one = np.ones(means.shape[1], dtype=np.int)
	for i in range(0, data.shape[0]):
		means[classification[i]] += data[i]
		count[classification[i]] += one
	for i in range(0, means.shape[0]):
		if count[i][0] == 0:
			count[i] += one
	return np.true_divide(means, count)

def draw(k, data, classification, iteration, dataset):
	color = iter(plt.cm.rainbow(np.linspace(0, 1, k)))
	plt.clf()
	for i in range(0, k):
		col = next(color)
		for j in range(0, data.shape[0]):
			if classification[j] == i:
				plt.scatter(data[j][0], data[j][1], s=8, c=col)
	title = "Spectral-Clustering Iteration-" + str(iteration)
	plt.suptitle(title)
	'''if dataset == "moon.txt":
		plt.savefig("./Screenshots/Spectral-Clustering/moon/" + title + ".png")
	else:
		plt.savefig("./Screenshots/Spectral-Clustering/circle/" + title + ".png")'''
	plt.show()

def draw_eigen_space(k, data, classification):
	color = iter(plt.cm.rainbow(np.linspace(0, 1, k)))
	plt.clf()
	title = "Spectral-Clustering in Eigen-Space"
	plt.suptitle(title)
	for i in range(0, k):
		col = next(color)
		for j in range(0, data.shape[0]):
			if classification[j] == i:
				plt.scatter(data[j][0], data[j][1], s=8, c=col)
	plt.savefig("./Screenshots/Spectral-Clustering/moon/" + title + ".png")
	plt.show()

def k_means(k, raw_data, data):
	# k is the number of cluster
	means, previous_classification, iteration = initialization(k, data) # means size: k*2 previous_classification: 3000
	classification = classify(data, means) # classification: 3000
	error = calculate_error(classification, previous_classification)
	#draw(k, raw_data, classification, iteration, dataset)
	while(True):
		iteration += 1
		means = update(data, means, classification)
		previous_classification = classification
		classification = classify(data, means)
		error = calculate_error(classification, previous_classification)
		#draw(k, raw_data, classification, iteration, dataset)
		print(error)
		if error < 5:
			break
	draw(k, raw_data, classification, iteration, dataset)
	return classification

if __name__ == "__main__":
	k = 2
	dataset = "circle.txt"
	data = read_input(dataset)
	Weight = compute_rbf_kernel(data) # Weight size: 3000 * 3000
	Degree = np.diag(np.sum(Weight, axis=1))
	Laplacian = Degree - Weight
	eigen_values, eigen_vectors = np.linalg.eig(Laplacian)
	idx = np.argsort(eigen_values)
	eigen_vectors = eigen_vectors[:,idx]
	U = (eigen_vectors[:,:k+1])[:,1:]
	classification = k_means(k, data, U)
	if k == 2:
		draw_eigen_space(k, U, classification)