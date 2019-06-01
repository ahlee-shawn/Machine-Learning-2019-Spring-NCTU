import matplotlib.pyplot as plt
import numpy as np
import csv
import time
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

def read_input(dataset):
	with open(dataset) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		data1 = list(csv_reader)
		data1 = [[float(y) for y in x] for x in data1]
	return np.array(data1).reshape(1500, 2)

def initialization(data, k):
	initialize_method = "randomly generate"
	print("Initialization Method: {}".format(initialize_method))
	previous_classification = np.zeros([1500], np.int)
	if initialize_method == "randomly generate":
		means = np.random.rand(k, 2)
	elif initialize_method == "randomly assign":
		temp = np.random.randint(low=0, high=data.shape[0], size=k)
		means = np.zeros([k, 2], dtype=np.float32)
		for i in range(0, k):
			means[i,:] = data[temp[i],:]
	elif initialize_method == "k-means++":
		means = np.zeros([k, 2], dtype=np.float32)
		temp = np.random.randint(low=0, high=data.shape[0], size=1, dtype=np.int)
		means[0,:] = data[temp,:]
		temp = np.zeros(data.shape[0], dtype=np.float32)
		for i in range(0, data.shape[0]):
			temp[i] = np.linalg.norm(data[i,:] - means[0,:])
		temp = temp / temp.sum()
		temp = np.random.choice(data.shape[0], 1, p=temp)
		means[1,:] = data[temp,:]
	return means, previous_classification, 1 # 1 for iteration

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
	return np.true_divide(means, count)

def draw(data, means, classification, iteration, dataset):
	color = iter(plt.cm.rainbow(np.linspace(0, 1, means.shape[0] * 2)))
	plt.clf()
	for i in range(0, means.shape[0]):
		col = next(color)
		for j in range(0, data.shape[0]):
			if classification[j] == i:
				plt.scatter(data[j][0], data[j][1], s=8, c=col)
	for i in range(0, means.shape[0]):
		col = next(color)
		plt.scatter(means[i][0], means[i][1], s=32, c=col)
	title = "K-Means Iteration-" + str(iteration)
	plt.suptitle(title)
	plt.show()
	'''
	if dataset == "moon.txt":
		plt.savefig("./Screenshots/K-Means/moon/" + title + ".png")
	else:
		plt.savefig("./Screenshots/K-Means/circle/" + title + ".png")
	'''

def k_means(data, dataset):
	# k is the number of cluster
	k = 2
	means, previous_classification, iteration = initialization(data, k) # means size: k*2 previous_classification: 3000
	classification = classify(data, means) # classification: 3000
	error = calculate_error(classification, previous_classification)
	#draw(data, means, classification, iteration, dataset)
	while(True):
		iteration += 1
		means = update(data, means, classification)
		previous_classification = classification
		classification = classify(data, means)
		error = calculate_error(classification, previous_classification)
		#draw(data, means, classification, iteration, dataset)
		if error < 3:
			break
	print("Elapsed Time: {}".format(time.time() - start_time))
	print("Iterations to coverged: {}".format(str(iteration)))
	#draw(data, means, classification, iteration, dataset)

if __name__ == "__main__":
	start_time = time.time()
	dataset = "moon.txt"
	print("Dataset: {}".format(dataset))
	data = read_input(dataset) # data size: 3000*2
	k_means(data, dataset)