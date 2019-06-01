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

def update(data, eps, i, j, classification, temp, min_points, current_cluster):
	classification[i] = current_cluster
	for p in range(0, min_points):
		classification[temp[p]] = current_cluster
	for p in range(j + 1, classification.shape[0]):
		if np.linalg.norm(data[i] - data[p]) < eps:
			if classification[p] == -1:
				classification[p] = current_cluster
			else:
				temp = classification[p]
				for q in range(0, classification.shape[0]):
					if classification[q] == temp:
						classification[q] = current_cluster
	return classification

def dbscan(data, eps, min_points, dataset):
	classification = np.full((data.shape[0]), -1)
	cluster_number = 0
	current_cluster = 0
	iteration = 0
	for i in range(0, data.shape[0]):
		iteration += 1
		if classification[i] == -1:
			current_cluster = cluster_number
		else:
			current_cluster = classification[i]
		data_in_eps = 0
		temp = np.zeros(min_points, dtype=np.int)
		for j in range(i + 1, data.shape[0]):
			if np.linalg.norm(data[i] - data[j]) < eps:
				if data_in_eps < min_points:
					temp[data_in_eps] = j
					data_in_eps += 1
					if data_in_eps == min_points:
						update(data, eps, i, j, classification, temp, min_points, current_cluster)
						cluster_number += 1
						break
		draw(data, classification, np.unique(classification), iteration, dataset)
	draw(data, classification, np.unique(classification), iteration, dataset)

def draw(data, classification, index, iteration, dataset):
	plt.clf()
	color = iter(plt.cm.rainbow(np.linspace(0, 1, index.shape[0])))
	for i in range(0, index.shape[0]):
		col = next(color)
		for j in range(0, classification.shape[0]):
			if classification[j] == -1:
				plt.scatter(data[j][0], data[j][1], s=8, c='black')
			elif classification[j] == index[i]:
				plt.scatter(data[j][0], data[j][1], s=8, c=col)
	title = "DBSCAN Iteration-" + str(iteration)
	plt.suptitle(title)
	if dataset == "moon.txt":
		plt.savefig("./Screenshots/DBSCAN/moon/" + title + ".png")
	else:
		plt.savefig("./Screenshots/DBSCAN/circle/" + title + ".png")

if __name__ == "__main__":
	eps = 0.1
	min_points = 10
	dataset = "moon.txt"
	data = read_input(dataset) # data size: 1500*2
	classification = dbscan(data, eps, min_points, dataset)