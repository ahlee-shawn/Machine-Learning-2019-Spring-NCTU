import sys
import csv
import numpy as np  
import matplotlib.pyplot as plt 
import math

def read_input(filename):
	x = []
	y = []
	with open(filename) as file:
		read = csv.reader(file, delimiter = ',')
		for row in read:
			x.append(float(row[0]))
			temp = []
			temp.append(float(row[1]))
			y.append(temp) #change y into 2D list
	return x, y

def matrix_A(x, polynomial_bases):
	A = []
	for i in range(len(x)):
		temp = []
		for j in range(polynomial_bases - 1, -1, -1):
			temp.append(x[i]**j)
		A.append(temp)
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

def matrix_add(A, B):
	result = []
	for i in range(len(A)):
		temp = []
		for j in range(len(A[0])):
			temp.append(A[i][j] + B[i][j])
		result.append(temp)
	return result

def LU_decomposition(A):
	L = I_mul_scalar(1, len(A))
	for i in range(len(A) - 1):
		L_inverse_temp = I_mul_scalar(1, len(A))
		L_temp = I_mul_scalar(1, len(A))
		for j in range(i + 1, len(A)):
			L_inverse_temp[j][i] = (-1) * A[j][i] / A[i][i]
			L_temp[j][i] = A[j][i] / A[i][i]
		A = matrix_mul(L_inverse_temp, A)
		L = matrix_mul(L_temp, L)
	return L, A

def print_LSE(result, A, y):
	print("LSE:")
	print("Fitting line:", end = "")
	i = 0
	formula = ""
	for j in range(len(result) - 1, -1, -1):
		print(" ", end = "")
		if result[i][0] >= 0:
			formula += str(result[i][0])
			print(result[i][0], end = " ")
		else:
			formula += str( -1 * result[i][0])
			print(-1 * result[i][0], end = " ")
		i += 1
		if j != 0:
			print("X^", end = "")
			print(j, end = " ")
			formula += ("*x**" + str(j)) 
			if result[i][0] >= 0:
				formula += "+"
				print("+", end = "")
			else:
				formula += "-"
				print("-", end = "")
	print("")
	predict = matrix_mul(A, result)
	error = 0.0
	for i in range(len(y)):
		error += (y[i][0] - predict[i][0]) ** 2
	print("Total error: ", error)
	return formula

def print_newton(result, A, y):
	print("\nNewton's Method:")
	print("Fitting line:", end = "")
	i = 0
	formula = ""
	for j in range(len(result) - 1, -1, -1):
		print(" ", end = "")
		if result[i][0] >= 0:
			formula += str(result[i][0])
			print(result[i][0], end = " ")
		else:
			formula += str( -1 * result[i][0])
			print(-1 * result[i][0], end = " ")
		i += 1
		if j != 0:
			print("X^", end = "")
			print(j, end = " ")
			formula += ("*x**" + str(j)) 
			if result[i][0] >= 0:
				formula += "+"
				print("+", end = "")
			else:
				formula += "-"
				print("-", end = "")
	print("")
	predict = matrix_mul(A, result)
	error = 0.0
	for i in range(len(y)):
		error += (y[i][0] - predict[i][0]) ** 2
	print("Total error: ", error)
	return formula

def graph(formula, X, Y):  
	plt.scatter(X, Y, c = 'red')
	x = np.array(range(math.floor(min(X)) - 1, math.ceil(max(X)) + 2))
	if 'x' in formula:
		y = np.array(eval(formula))
		plt.plot(x, y)  
		plt.show()
	else:
		plt.axhline(y = float(formula), color='b', linestyle='-') 
		plt.show()

def L_mul_y_equal_b(L, b):
	y_temp = I_mul_scalar(0, len(L))
	return y_temp

if __name__ == "__main__":
	filename = sys.argv[1]
	polynomial_bases = int(sys.argv[2])
	lse_lambda = float(sys.argv[3])
	x, y = read_input(filename)
	A = matrix_A(x, polynomial_bases)
	A_transpose = matrix_transpose(A)
	A_transpose_mul_A = matrix_mul(A_transpose, A)
	# LSE
	lambda_I = I_mul_scalar(lse_lambda, len(A_transpose_mul_A))
	A_transpose_mul_A_add_lambda_I = matrix_add(A_transpose_mul_A, lambda_I)
	L, U = LU_decomposition(A_transpose_mul_A_add_lambda_I)
	print(A_transpose_mul_A_add_lambda_I)
	print(L)
	print(U)
	y_temp = L_mul_y_equal_b(L, I_mul_scalar(1, len(L)))
	'''
	#U_inverse = upper_matrix_inverse(U)
	A_transpose_mul_A_add_lambda_I_inverse = matrix_mul(U_inverse, L_inverse)
	A_transpose_mul_b = matrix_mul(A_transpose, y)
	result = matrix_mul(A_transpose_mul_A_add_lambda_I_inverse, A_transpose_mul_b)
	formula_lse = print_LSE(result, A, y)

	# Newton
	L_inverse, U = LU_decomposition(A_transpose_mul_A)
	U_inverse = upper_matrix_inverse(U)
	A_transpose_mul_A_inverse = matrix_mul(U_inverse, L_inverse)
	result = matrix_mul(A_transpose_mul_A_inverse, A_transpose_mul_b)
	formula_newton = print_newton(result, A, y)

	# Visualization
<<<<<<< HEAD
	graph(formula_lse, formula_newton, x, y)
	'''
=======
	graph(formula_lse, x, y)
	graph(formula_newton, x, y)
>>>>>>> parent of 8dbba66... merge tow plots into onr figure with two subplots
