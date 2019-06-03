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

def calculate_neighbor(data, eps, min_points):
	result = np.zeros(data.shape[0], dtype=np.int)
	for i in range(0, data.shape[0]):
		temp = 0
		for j in range(0, data.shape[0]):
			if np.linalg.norm(data[i] - data[j]) <= eps:
				temp += 1
			if temp == min_points:
				result[i] = 1
				break
	return result

def update(data, eps, i, classification, neighbor, iteration, cluster_number, dataset):
	if classification[i] == -1:
		uncheck_list = []
		uncheck_list.append(i)
		classification[i] = cluster_number
		draw(data, classification, np.unique(classification), iteration, dataset)
		while(len(uncheck_list)):
			i = uncheck_list.pop(0)
			for j in range(0, data.shape[0]):
				if classification[j] == -1:
					if np.linalg.norm(data[i] - data[j]) <= eps and classification[j] == -1:
						classification[j] = cluster_number
						uncheck_list.append(j)
		iteration += 1
		cluster_number += 1
	return classification, iteration, cluster_number

def dbscan(data, eps, min_points, dataset):
	neighbor = calculate_neighbor(data, eps, min_points)
	classification = np.full((data.shape[0]), -1)
	cluster_number = 0
	iteration = 0
	for i in range(0, data.shape[0]):
		if neighbor[i] != 0:
			classification, iteration, cluster_number = update(data, eps, i, classification, neighbor, iteration, cluster_number, dataset)
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
	#plt.show()

if __name__ == "__main__":
	eps = 0.1
	min_points = 10
	dataset = "circle.txt"
	data = read_input(dataset) # data size: 1500*2
	classification = dbscan(data, eps, min_points, dataset)