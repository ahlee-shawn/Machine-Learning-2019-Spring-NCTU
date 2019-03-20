import sys
import numpy as np

def read_train_discrete():
	prior = np.zeros((10), dtype = int)
	image_bin = np.zeros((10, 28 * 28, 32), dtype = int)

	file_train_image = open("train-images-idx3-ubyte", 'rb')
	file_train_label = open("train-labels-idx1-ubyte", 'rb')
	
	file_train_image.read(4) # magic number in image training file
	file_train_image.read(4) # number of images in training image file
	file_train_image.read(4) # number of rows in training image file
	file_train_image.read(4) # number of columns in training image file
	file_train_label.read(4) # magic number in training label file
	file_train_label.read(4) # number of items in training label file

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

def print_imagination(image_bin):
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
	file_test_image = open("t10k-images-idx3-ubyte", 'rb')
	file_test_label = open("t10k-labels-idx1-ubyte", 'rb')

	file_test_image.read(4) # magic number in image testing file
	file_test_image.read(4) # number of images in testing image file
	file_test_image.read(4) # number of rows in testing image file
	file_test_image.read(4) # number of columns in testing image file
	file_test_label.read(4) # magic number in testing label file
	file_test_label.read(4) # number of items in testing label file

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
				if temp <= 1:
					probability[j] += np.log(float(1 / image_bin_sum[j][k]))
				else:
					probability[j] += np.log(float(image_bin[j][k][test_image[k]] / image_bin_sum[j][k]))
		temp = 0
		for j in range(10):
			temp += probability[j]
		for j in range(10):
			probability[j] /= temp
		error += print_result(probability, answer)
	print_imagination(image_bin)
	print("Error rate: ", float(error / 10000))

def read_train_continuous():
	return

if __name__ == "__main__":
	if sys.argv[1] == '0':
		image_bin, prior = read_train_discrete()
		test_discrete(image_bin, prior)
	else:
		read_train_continuous()