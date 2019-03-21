import numpy as np
from sklearn.naive_bayes import GaussianNB

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

def train_continuous():
	file_train_image, file_train_label = open_train_file()
	X = np.zeros((60000, 28 * 28), dtype = float)
	Y = np.zeros((60000), dtype = float)
	for i in range(60000):
		for j in range(28 * 28):
			X[i][j] = int.from_bytes(file_train_image.read(1), byteorder = 'big')
		Y[i] = int.from_bytes(file_train_label.read(1), byteorder = 'big')
	clf = GaussianNB()
	clf.fit(X, Y)
	file_test_image, file_test_label = open_test_file()
	error = 0
	for i in range(10000):
		test_image = np.zeros((1, 28 * 28), dtype = float)
		for j in range(28 * 28):
			test_image[0][j] = int.from_bytes(file_test_image.read(1), byteorder = 'big')
		label = int.from_bytes(file_test_label.read(1), byteorder = 'big')
		prediction = clf.predict(test_image)
		if label != prediction[0]:
			error += 1
	print(error)

if __name__ == "__main__":
	train_continuous()