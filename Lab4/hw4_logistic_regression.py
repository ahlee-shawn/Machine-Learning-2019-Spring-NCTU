import numpy as np
import math
import matplotlib.pyplot as plt

def matrix_transpose(x):
	transpose = []
	for i in range(len(x[0])):
		temp = []
		for j in range(len(x)):
			temp.append(x[j][i])
		transpose.append(temp)
	return transpose

def matrix_mul(A, B):
	result = []
	for i in range(len(A)):
		temp = []
		for j in range(len(B[0])):
			temp.append(0)
		result.append(temp)
	for i in range(len(A)):
		for j in range(len(B[0])):
			for k in range(len(B)):
				result[i][j] += A[i][k] * B[k][j]
	return result

def matrix_minus(A, B):
	result = []
	for i in range(len(A)):
		temp = []
		for j in range(len(A[0])):
			temp.append(A[i][j] - B[i][j])
		result.append(temp)
	return result

def update(A, B):
	result = []
	rate = 1
	for i in range(len(A)):
		temp = []
		for j in range(len(A[0])):
			temp.append(A[i][j] + B[i][j] * rate)
		result.append(temp)
	return result

def get_input():
	n = int(input("Number of data points: "))
	mx1 = float(input("mx1: "))
	my1 = float(input("my1: "))
	mx2 = float(input("mx2: "))
	my2 = float(input("my2: "))
	vx1 = float(input("vx1: "))
	vy1 = float(input("vy1: "))
	vx2 = float(input("vx2: "))
	vy2 = float(input("vy2: "))
	return n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2

def univariate_gaussian_data_generator(mean, standard_deviation):
	temp = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
	return mean + standard_deviation * temp

def get_data(n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2):
	c1x = []
	c1y = []
	c2x = []
	c2y = []
	X = []
	y =[]
	for i in range(0, n):
		temp1 = []
		x1 = univariate_gaussian_data_generator(mx1, math.sqrt(vx1))
		y1 = univariate_gaussian_data_generator(my1, math.sqrt(vy1))
		c1x.append(x1)
		c1y.append(y1)
		temp1.append(x1)
		temp1.append(y1)
		temp1.append(1.0)
		X.append(temp1)
		y.append([0.0])
		
		temp2 = []
		x2 = univariate_gaussian_data_generator(mx2, math.sqrt(vx2))
		y2 = univariate_gaussian_data_generator(my2, math.sqrt(vy2))
		c2x.append(x2)
		c2y.append(y2)
		temp2.append(x2)
		temp2.append(y2)
		temp2.append(1.0)
		X.append(temp2)
		y.append([1.0])
	return X, y, c1x, c1y, c2x, c2y

def sigmoid(A):
	matrix = []
	for i in range(0, len(A)):
		temp = []
		for j in range(0, len(A[0])):
			temp.append(1.0 / (1.0 + np.exp(-1.0 * A[i][j])))
		matrix.append(temp)
	return matrix

def difference(A, B):
	temp = True
	for i in range(0, len(A)):
		if abs(A[i][0] - B[i][0]) > (abs(B[i][0]) * 0.05):
			temp = False
			break
	return temp

def draw(X, c1x, c1y, c2x, c2y, gradient_w, y):
	plt.subplot(131)
	plt.title("Ground Truth")
	plt.scatter(c1x, c1y, c = 'r')
	plt.scatter(c2x, c2y, c = 'b')

	print("Gradient Descent:\n")
	confusion_matrix = [[0, 0], [0, 0]]
	predict = sigmoid(matrix_mul(X, gradient_w))
	c1x = []
	c1y = []
	c2x = []
	c2y = []
	for i in range(0, len(predict)):
		if predict[i][0] < 0.5:
			c1x.append(X[i][0])
			c1y.append(X[i][1])
		else:
			c2x.append(X[i][0])
			c2y.append(X[i][1])
	for i in range(0, len(predict)):
		if y[i][0] == 0:
			if predict[i][0] < 0.5:
				confusion_matrix[0][0] += 1
			else:
				confusion_matrix[0][1] += 1
		if y[i][0] == 1:
			if predict[i][0] < 0.5:
				confusion_matrix[1][0] += 1
			else:
				confusion_matrix[1][1] += 1

	print("w:")
	for i in range(0, len(gradient_w)):
		print(gradient_w[i][0])
	print("\nconfusion_matrix")
	print("\t\t Predict cluster 1 Predict cluster 2")
	print("Is cluster 1\t\t {}\t\t{}" .format(confusion_matrix[0][0], confusion_matrix[0][1]))
	print("Is cluster 2\t\t {}\t\t{}\n" .format(confusion_matrix[1][0], confusion_matrix[1][1]))
	print("Sensitivity (Successfully predict cluster 1): {}" .format(confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])))
	print("Specificity (Successfully predict cluster 2): {}" .format(confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])))

	plt.subplot(132)
	plt.title("Gradient Descent")
	plt.scatter(c1x, c1y, c = 'r')
	plt.scatter(c2x, c2y, c = 'b')




	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2 = get_input()
	X, y, c1x, c1y, c2x, c2y = get_data(n, mx1, vx1, my1, vy1, mx2, vx2, my2, vy2)
	w =[[0.0], [0.0], [0.0]]
	new_w = [[0.0], [0.0], [0.0]]
	while(True):
		sigmoid_input = matrix_mul(X, w)
		partial_derivative = matrix_mul(matrix_transpose(X), matrix_minus(y, sigmoid(sigmoid_input)))
		new_w = update(w, partial_derivative)
		if difference(new_w, w):
			break
		w = new_w
	gradient_w = w
	draw(X, c1x, c1y, c2x, c2y, gradient_w, y)















