import matplotlib.pyplot as plt
import numpy as np
import csv

def read_input():
	with open('moon.txt') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		data1 = list(csv_reader)
		data1 = [[float(y) for y in x] for x in data1]
	with open('circle.txt') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		data2 = list(csv_reader)
		data2 = [[float(y) for y in x] for x in data2]
	return np.append(np.array(data1), np.array(data2)).reshape(3000, 2)

def initialization(k):
	means = np.random.rand(k, 2)
	previous_classification = np.zeros([3000], np.int)
	return means, previous_classification

def classify(data, means):
	classification = np.zeros([3000], dtype=int)
	for i in range(0, 3000):
		temp = np.zeros([means.shape[0]], dtype=np.float32) # temp size: k
		for j in range(0, means.shape[0]):
			temp[j] = np.linalg.norm(data[i] - means[j])
		classification[i] = np.argmin(temp)
	return classification

def calculate_error(classification, previous_classification):
	error = 0
	for i in range(0, 3000):
		error += np.absolute(classification[i] - previous_classification[i])
	return error

def update(data, means, classification):
	means = np.zeros(means.shape, dtype=np.float32)
	count = np.zeros(means.shape, dtype=int)
	for i in range(0, 3000):
		means[classification[i]] += data[i]
		count[classification[i]] += 1
	return np.true_divide(means, count)

def draw(data, means, classification):
	plt.title("K-Means")
	colors = ['g', 'b']
	for i in range(0, means.shape[0]):
		for j in range(0, data.shape[0]):
			if classification[j] == i:
				plt.scatter(data[j][0], data[j][1], s=8, c=colors[i])
	plt.scatter(means[:,0], means[:,1], s=32, c='r')
	plt.show()

def k_means(data):
	# k is the number of cluster
	k = 2
	means, previous_classification = initialization(k) # temp size: k*2 previous_classification: 3000
	classification = classify(data, means) # classification: 3000
	error = calculate_error(classification, previous_classification)
	draw(data, means, classification)
	while(error):
		means = update(data, means, classification)
		previous_classification = classification
		classification = classify(data, means)
		error = calculate_error(classification, previous_classification)
		draw(data, means, classification)
		print(error)

if __name__ == "__main__":
	data = read_input() # data size: 3000*2
	k_means(data)