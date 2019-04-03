import numpy as np
import math

def univariate_gaussian_data_generator(mean, standard_deviation):
	temp = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
	return mean + standard_deviation * temp

def polynomial_basis_linear_model_data_generator(n, standard_deviation, w):
	x = float(np.random.uniform(-1.0, 1.0, 1))
	y = 0.0
	for i in range(n):
		y += w[0][i] * (x ** i)
	e = float(univariate_gaussian_data_generator(0, standard_deviation))
	y += e
	return y

if __name__ == "__main__":

	# univariate gaussian data generator
	print("Mean: ", end="")
	mean = float(input())
	print("Variance: ", end = "")
	variance = float(input())
	standard_deviation = math.sqrt(variance)
	print(univariate_gaussian_data_generator(mean, standard_deviation))

	# polynomial basis linear model data generator
	print("N: ", end = "")
	n = int(input())
	print("a: ", end = "")
	variance = float(input())
	standard_deviation = math.sqrt(variance)
	w = [[]]
	print("w: ", end = "")
	for i in range(n):
		w[0].append(float(input()))
	print(polynomial_basis_linear_model_data_generator(n, standard_deviation, w))