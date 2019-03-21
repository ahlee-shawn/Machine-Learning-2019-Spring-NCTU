import sys
import numpy as np
import math

def open_train_file():
	file_train_image = open("train-images-idx3-ubyte", 'rb')
	file_train_label = open("train-labels-idx1-ubyte", 'rb')

	file_train_image.read(4) # magic number in image training file
	file_train_image.read(4) # number of images in training image file
	file_train_image.read(4) # number of rows in training image file
	file_train_image.read(4) # number of columns in training image file
	file_train_label.read(4) # magic number in training label file
	file_train_label.read(4) # number of items in training label file

	return file_train_image, file_train_label

def open_test_file():
	file_test_image = open("t10k-images-idx3-ubyte", 'rb')
	file_test_label = open("t10k-labels-idx1-ubyte", 'rb')

	file_test_image.read(4) # magic number in image testing file
	file_test_image.read(4) # number of images in testing image file
	file_test_image.read(4) # number of rows in testing image file
	file_test_image.read(4) # number of columns in testing image file
	file_test_label.read(4) # magic number in testing label file
	file_test_label.read(4) # number of items in testing label file

	return file_test_image, file_test_label

def read_train_discrete():
	prior = np.zeros((10), dtype = int)
	image_bin = np.zeros((10, 28 * 28, 32), dtype = int)

	file_train_image, file_train_label = open_train_file()

	for i in range(60000):
		label = int.from_bytes(file_train_label.read(1), byteorder = 'big')
		prior[label] += 1
		for j in range(28 * 28):
			pixel = int.from_bytes(file_train_image.read(1), byteorder = 'big')
			image_bin[label][j][int(pixel / 8)] += 1

	return image_bin, prior

def preprocess(image_bin):
	image_bin_sum = np.zeros((10, 784), dtype = int)
	for i in range(10):
		for j in range(28 * 28):
			for k in range(32):
				image_bin_sum[i][j] += image_bin[i][j][k]
	return image_bin_sum

def normalization(probability):
	temp = 0
	for j in range(10):
		temp += probability[j]
	for j in range(10):
		probability[j] /= temp
	return probability

def print_result(probability, answer):
	print("Posterior (in log scale):")
	for j in range(10):
		print(j, ": ", probability[j])
	prediction = np.argmin(probability)
	print("Prediction: ", prediction, ", Ans: ", answer)
	print("")
	if prediction == answer:
		return 0
	else:
		return 1

def print_imagination_discrete(image_bin):
	print("Imagination of numbers in Bayesian Classifier:")
	print("")
	for i in range(10):
		print(i, ":")
		for j in range(28):
			for k in range(28):
				temp = 0
				for t in range(16):
					temp += image_bin[i][j * 28 + k][t]
				for t in range(16, 32):
					temp -= image_bin[i][j * 28 + k][t]
				if temp > 0:
					print("0", end = " ")
				else:
					print("1", end = " ")
			print("")
		print("")

def test_discrete(image_bin, prior):
	file_test_image, file_test_label = open_test_file()

	image_bin_sum = preprocess(image_bin)

	error = 0
	for i in range(10000):
		answer = int.from_bytes(file_test_label.read(1), byteorder = 'big')
		probability = np.zeros((10), dtype = float)
		test_image = np.zeros((28 * 28), dtype = int)
		for j in range(28 * 28):
			test_image[j] = int((int.from_bytes(file_test_image.read(1), byteorder = 'big')) / 8)
		for j in range(10):
			probability[j] += np.log(float(prior[j] / 60000))
			for k in range(28 * 28):
				temp = image_bin[j][k][test_image[k]]
				if temp == 0:
					probability[j] += np.log(float(0.001 / image_bin_sum[j][k]))
				else:
					probability[j] += np.log(float(image_bin[j][k][test_image[k]] / image_bin_sum[j][k]))
		probability = normalization(probability)
		error += print_result(probability, answer)
	print_imagination_discrete(image_bin)
	print("Error rate: ", float(error / 10000))

def read_train_continuous():
	prior = np.zeros((10), dtype = int)
	pixel_square = np.zeros((10, 28 * 28), dtype = float)
	mean = np.zeros((10, 28 * 28), dtype = float)
	var = np.zeros((10, 28 * 28), dtype = float)
	
	file_train_image, file_train_label = open_train_file()

	for i in range(60000):
		label = int.from_bytes(file_train_label.read(1), byteorder = 'big')
		prior[label] += 1
		for j in range(28 * 28):
			pixel = int.from_bytes(file_train_image.read(1), byteorder = 'big')
			mean[label][j] += pixel
			pixel_square[label][j] += (pixel ** 2)

	for i in range(10):
		for j in range(28 * 28):
			mean[i][j] = float(mean[i][j] / prior[i])
			pixel_square[i][j] = float(pixel_square[i][j] / prior[i])
			var[i][j] = pixel_square[i][j] - (mean[i][j] ** 2)

	for i in range(10):
		for j in range(28 * 28):
			if var[i][j] == 0:
				var[i][j] = 0.0001

	return mean, var, prior

def print_imagination_continuous(mean):
	print("Imagination of numbers in Bayesian Classifier:")
	print("")
	for i in range(10):
		print(i, ":")
		for j in range(28):
			for k in range(28):
				if mean[i][j * 28 + k] < 128:
					print("0", end = " ")
				else:
					print("1", end = " ")
			print("")
		print("")

def test_continuous(mean, var, prior):
	file_test_image, file_test_label = open_test_file()

	error = 0
	for i in range(10000):
		answer = int.from_bytes(file_test_label.read(1), byteorder = 'big')
		probability = np.zeros((10), dtype = float)
		test_image = np.zeros((28 * 28), dtype = float)
		for j in range(28 * 28):
			test_image[j] = int.from_bytes(file_test_image.read(1), byteorder = 'big')
		for j in range(10):
			probability[j] += np.log(float(prior[j] / 60000))
			for k in range(28 * 28):
				probability[j] += np.log(float(1.0 / math.sqrt(2.0 * math.pi * var[j][k])))
				probability[j] -=  float(((test_image[k] - mean[j][k]) ** 2) / (2 * var[j][k]))
		probability = normalization(probability)
		error += print_result(probability, answer)
	print_imagination_continuous(mean)
	print("Error rate: ", float(error / 10000))

if __name__ == "__main__":
	if sys.argv[1] == '0':
		image_bin, prior = read_train_discrete()
		test_discrete(image_bin, prior)
	else:
		mean, var, prior = read_train_continuous()
		test_continuous(mean, var, prior)