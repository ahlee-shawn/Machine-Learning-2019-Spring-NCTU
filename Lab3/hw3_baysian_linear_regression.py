import numpy as np
import math

def polynomial_basis_linear_model_data_generator(n, standard_deviation, w):
	x = float(np.random.uniform(-1.0, 1.0, 1))
	y = 0.0
	for i in range(n):
		y += w[i] * (x ** i)
	e = float(univariate_gaussian_data_generator(0, standard_deviation))
	y += e
	return y

if __name__ == "__main__":

	# polynomial basis linear model data generator
	print("N: ", end = "")
	n = int(input())
	print("a: ", end = "")
	variance = float(input())
	standard_deviation = math.sqrt(variance)
	w = []
	print("w: ", end = "")
	for i in range(n):
		w.append(float(input()))
	print(polynomial_basis_linear_model_data_generator(n, standard_deviation, w))