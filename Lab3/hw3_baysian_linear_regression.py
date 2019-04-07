import numpy as np
import math
import matplotlib.pyplot as plt

def get_input():
	print("The precision for initial prior: ", end = "")
	b = float(input())
	print("n: ", end = "")
	n = int(input())
	print("a: ", end = "")
	variance_error = float(input())
	standard_deviation_error = math.sqrt(variance_error)
	print("w: ", end = "")
	w = input().split()
	for i in range(len(w)):
		w[i] = float(w[i])
	return b, n, variance_error, standard_deviation_error, w

def univariate_gaussian_data_generator(mean, standard_deviation):
	temp = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
	return mean + standard_deviation * temp

def polynomial_basis_linear_model_data_generator(n, standard_deviation_error, w):
	x = float(np.random.uniform(-1.0, 1.0, 1))
	y = 0.0
	for i in range(n):
		y += w[i] * (x ** i)
	return x, y + float(univariate_gaussian_data_generator(0, standard_deviation_error))

def matrix_A(x, polynomial_bases):
	A = [[]]
	for j in range(0, polynomial_bases):
		A[0].append(x ** j)
	return A

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

def matrix_add(A, B):
	result = []
	for i in range(len(A)):
		temp = []
		for j in range(len(A[0])):
			temp.append(A[i][j] + B[i][j])
		result.append(temp)
	return result

def matrix_mul_scalar(A, scalar):
	result = []
	for i in range(len(A)):
		temp = []
		for j in range(len(A[0])):
			temp.append(A[i][j] * scalar)
		result.append(temp)
	return result

def I_mul_scalar(scalar, size):
	result = []
	for i in range(size):
		temp = []
		for j in range(size):
			if(i == j):
				temp.append(scalar)
			else:
				temp.append(0)
		result.append(temp)
	return result

def LU_decomposition(A):
	L_inverse = I_mul_scalar(1, len(A))
	for i in range(len(A) - 1):
		L_temp = I_mul_scalar(1, len(A))
		for j in range(i + 1, len(A)):
			L_temp[j][i] = (-1) * A[j][i] / A[i][i]
		A = matrix_mul(L_temp, A)
		L_inverse = matrix_mul(L_temp, L_inverse)
	return L_inverse, A

def upper_matrix_inverse(U, L_inverse):
	result = I_mul_scalar(0, len(U))
	for i in range(len(U)-1, -1, -1):
		for j in range(len(result[0])):
			temp = 0
			for k in range(len(result)):
				temp += U[i][k] * result[k][j]
			result[i][j] = (L_inverse[i][j] - temp) / U[i][i]
	return result

def difference(list1, list2):
	temp = 0
	for i in range(0, len(list1)):
		temp += ((list1[i][0] - list2[i][0]) ** 2)
	return math.sqrt(temp)

def draw(w, variance_error, data_x, data_y, m, n, ten_data_x, ten_data_y, ten_incomes_m, ten_a, ten_S_inverse, fifty_data_x, fifty_data_y, fifty_incomes_m, fifty_a, fifty_S_inverse):
	plt.subplot(221)
	plt.xlim(-2.0, 2.0)
	plt.title("Ground Truth")
	ground_func = np.poly1d(np.flip(w))
	ground_x = np.linspace(-2.0, 2.0, 30)
	ground_y = ground_func(ground_x)
	print(type(ground_y))
	plt.plot(ground_x, ground_y, color = 'black')
	w[0] += variance_error
	ground_func = np.poly1d(np.flip(w))
	ground_y_1 = ground_func(ground_x)
	plt.plot(ground_x, ground_y_1, color = 'red')
	w[0] -= 2 * variance_error
	ground_func = np.poly1d(np.flip(w))
	ground_y_2 = ground_func(ground_x)
	plt.plot(ground_x, ground_y_2, color = 'red')
	plt.ylim(min(ground_y_2) - 1.0, max(ground_y_1) + 1.0)

	plt.subplot(222)
	plt.xlim(-2.0, 2.0)
	plt.ylim(min(ground_y_2) - 1.0, max(ground_y_1) + 1.0)
	plt.title("Predict Result")
	plt.scatter(data_x, data_y)
	predict_func = np.poly1d(np.flip(np.reshape(m, n)))
	predict_x = np.linspace(-2.0, 2.0, 30)
	predict_y = predict_func(predict_x)
	plt.plot(predict_x, predict_y, color = 'black')
	predict_y_add_variance = []
	for i in range(0, 30):
		A = matrix_A(predict_x[i], n)
		A_transpose = matrix_transpose(A)
		predict_y_add_variance.append(predict_y[i] + ((1.0/a) + (matrix_mul(A, matrix_mul(S_inverse, A_transpose))[0][0])))
	plt.plot(predict_x, predict_y_add_variance, color = 'red')

	predict_y_minus_variance = []
	for i in range(0, 30):
		A = matrix_A(predict_x[i], n)
		A_transpose = matrix_transpose(A)
		predict_y_minus_variance.append(predict_y[i] - ((1.0/a) + (matrix_mul(A, matrix_mul(S_inverse, A_transpose))[0][0])))
	plt.plot(predict_x, predict_y_minus_variance, color = 'red')

	plt.subplot(223)
	plt.xlim(-2.0, 2.0)
	plt.ylim(min(ground_y_2) - 1.0, max(ground_y_1) + 1.0)
	plt.title("After 10 incomes")
	plt.scatter(ten_data_x, ten_data_y)
	ten_func = np.poly1d(np.flip(np.reshape(ten_incomes_m, n)))
	ten_data_x = np.linspace(-2.0, 2.0, 30)
	ten_data_y = ten_func(ten_data_x)
	plt.plot(ten_data_x, ten_data_y, color = 'black')
	ten_y_add_variance = []
	for i in range(0, 30):
		A = matrix_A(ten_data_x[i], n)
		A_transpose = matrix_transpose(A)
		ten_y_add_variance.append(ten_data_y[i] + ((1.0/ten_a) + (matrix_mul(A, matrix_mul(ten_S_inverse, A_transpose))[0][0])))
	plt.plot(ten_data_x, ten_y_add_variance, color = 'red')

	ten_y_minus_variance = []
	for i in range(0, 30):
		A = matrix_A(ten_data_x[i], n)
		A_transpose = matrix_transpose(A)
		ten_y_minus_variance.append(ten_data_y[i] - ((1.0/ten_a) + (matrix_mul(A, matrix_mul(ten_S_inverse, A_transpose))[0][0])))
	plt.plot(ten_data_x, ten_y_minus_variance, color = 'red')


	plt.subplot(224)
	plt.xlim(-2.0, 2.0)
	plt.ylim(min(ground_y_2) - 1.0, max(ground_y_1) + 1.0)
	plt.title("After 50 incomes")
	plt.scatter(fifty_data_x, fifty_data_y)
	fifty_func = np.poly1d(np.flip(np.reshape(fifty_incomes_m, n)))
	fifty_data_x = np.linspace(-2.0, 2.0, 30)
	fifty_data_y = fifty_func(fifty_data_x)
	plt.plot(fifty_data_x, fifty_data_y, color = 'black')
	fifty_y_add_variance = []
	for i in range(0, 30):
		A = matrix_A(fifty_data_x[i], n)
		A_transpose = matrix_transpose(A)
		fifty_y_add_variance.append(fifty_data_y[i] + ((1.0/fifty_a) + (matrix_mul(A, matrix_mul(fifty_S_inverse, A_transpose))[0][0])))
	plt.plot(fifty_data_x, fifty_y_add_variance, color = 'red')

	fifty_y_minus_variance = []
	for i in range(0, 30):
		A = matrix_A(fifty_data_x[i], n)
		A_transpose = matrix_transpose(A)
		fifty_y_minus_variance.append(fifty_data_y[i] - ((1.0/fifty_a) + (matrix_mul(A, matrix_mul(fifty_S_inverse, A_transpose))[0][0])))
	plt.plot(fifty_data_x, fifty_y_minus_variance, color = 'red')

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":

	b, n, variance_error, standard_deviation_error, w = get_input()

	m = []
	prev_m = []
	for i in range(0, n):
		prev_m.append([-10.0])
		m.append([0.0])
	data_mean = 0.0
	data_variance = 0.0
	prev_data_mean = 0.0
	data_x = []
	data_y = []
	k = 0

	while(True):
		print("--------------------------------------------------------------")
		k += 1
		new_data_y = [[0.0]]
		new_data_x, new_data_y[0][0] = polynomial_basis_linear_model_data_generator(n, standard_deviation_error, w)
		data_x.append(new_data_x)
		data_y.append(new_data_y[0][0])
		
		data_mean = (data_mean * (k - 1) + new_data_y[0][0]) / k
		data_variance = data_variance + (prev_data_mean ** 2) - (data_mean ** 2) + (((new_data_y[0][0] ** 2) - data_variance - (prev_data_mean ** 2)) / (k + 1))
		if data_variance == 0:
			a = 0.0001
		else:
			a = data_variance
		
		A = matrix_A(new_data_x, n)
		A_transpose = matrix_transpose(A)
		A_transpose_mul_A = matrix_mul(A_transpose, A)
		if k == 1:
			sigma_inverse = matrix_add(matrix_mul_scalar(A_transpose_mul_A, a), I_mul_scalar(b, n))
			L_inverse, U = LU_decomposition(sigma_inverse)
			S_inverse = upper_matrix_inverse(U, L_inverse)
			m = matrix_mul_scalar(matrix_mul(S_inverse, matrix_mul(A_transpose, new_data_y)), a)
		else:
			L_inverse, U = LU_decomposition(S_inverse)
			S = upper_matrix_inverse(U, L_inverse)

			sigma_inverse = matrix_add(matrix_mul_scalar(A_transpose_mul_A, a), S)
			L_inverse, U = LU_decomposition(sigma_inverse)
			S_inverse = upper_matrix_inverse(U, L_inverse)

			a_mul_A_tranpose_mul_b = matrix_mul_scalar(matrix_mul(A_transpose, new_data_y), a)
			m = matrix_mul(S_inverse, matrix_add(a_mul_A_tranpose_mul_b, matrix_mul(S, m)))

		predictive_distribution_mean = matrix_mul(A, m)[0][0]
		predictive_distribution_variance = (1.0 / a) + (matrix_mul(A, matrix_mul(S_inverse, A_transpose))[0][0])
		
		print("Add data point (", new_data_x, ", ", new_data_y[0][0], ")")
		print("Posterior mean: ")
		for i in range(0, len(m)):
			print(m[i][0])
		print("")
		print("Posterior variance: ")
		for i in range(0, len(S_inverse)):
			for j in range(0, len(S_inverse)):
				if j != (len(S_inverse) - 1):	
					print(S_inverse[i][j], end = ", ")
				else:
					print(S_inverse[i][j])
		print("")
		print("Predictive Distribution ~ N ({0}, {1})".format(predictive_distribution_mean, predictive_distribution_variance))

		if difference(prev_m, m) < 0.01 and k > 1000:
			break
		prev_m = m
		prev_data_mean = data_mean
		if k == 10:
			ten_incomes_m = m.copy()
			ten_data_x = data_x.copy()
			ten_data_y = data_y.copy()
			ten_a = a
			ten_S_inverse = S_inverse.copy()
		if k == 50:
			fifty_incomes_m = m.copy()
			fifty_data_x = data_x.copy()
			fifty_data_y = data_y.copy()
			fifty_a = a
			fifty_S_inverse = S_inverse.copy()

draw(w, variance_error, data_x, data_y, m, n, ten_data_x, ten_data_y, ten_incomes_m, ten_a, ten_S_inverse, fifty_data_x, fifty_data_y, fifty_incomes_m, fifty_a, fifty_S_inverse)