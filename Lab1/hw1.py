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

def matrix_minus(A, B):
	result = []
	for i in range(len(A)):
		temp = []
		for j in range(len(A[0])):
			temp.append(A[i][j] - B[i][j])
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

def difference(A):
	temp = 0
	for i in range(len(A)):
		for j in range(len(A[0])):
			temp += (A[i][j])**2
	return math.sqrt(temp)

def print_error(method, result, A, y):
	print(method)
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
			if i != 0:
				print(-1 * result[i][0], end = " ")
			else:
				print(result[i][0], end = " ")
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

def graph(formula_lse, formula_newton, X, Y):  
	plt.subplot(2,1,1)
	plt.title("LSE")
	plt.scatter(X, Y, c = 'red')
	x = np.array(range(math.floor(min(X)) - 1, math.ceil(max(X)) + 2))
	if 'x' in formula_lse:
		y = np.array(eval(formula_lse))
		plt.plot(x, y)  
	else:
		plt.axhline(y = float(formula_lse), color='b', linestyle='-') 
	plt.subplot(2,1,2)
	plt.tight_layout()
	plt.title("Newton's Method")
	plt.scatter(X, Y, c = 'red')
	if 'x' in formula_newton:
		y = np.array(eval(formula_newton))
		plt.plot(x, y)  
		plt.show()
	else:
		plt.axhline(y = float(formula_newton), color='b', linestyle='-') 
		plt.show()

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
	L_inverse, U = LU_decomposition(A_transpose_mul_A_add_lambda_I)
	A_transpose_mul_A_add_lambda_I_inverse = upper_matrix_inverse(U, L_inverse)
	A_transpose_mul_b = matrix_mul(A_transpose, y)
	result = matrix_mul(A_transpose_mul_A_add_lambda_I_inverse, A_transpose_mul_b)
	formula_lse = print_error("LSE:", result, A, y)
	
	# Newton
	result = []
	for i in range(polynomial_bases):
		result.append([100]) 
	L_inverse, U = LU_decomposition(matrix_mul_scalar(A_transpose_mul_A, 2))
	hessian_matrix_inverse = upper_matrix_inverse(U, L_inverse)
	two_mul_A_transpose_mul_b = matrix_mul_scalar(matrix_mul(A_transpose, y), 2)
	while(1):
		temp = result
		gradient = matrix_minus(matrix_mul_scalar(matrix_mul(A_transpose_mul_A, result), 2), two_mul_A_transpose_mul_b)
		result = matrix_minus(result, matrix_mul(hessian_matrix_inverse, gradient))
		if difference(matrix_minus(result, temp)) < polynomial_bases:
			break
	formula_newton = print_error("\nNewton's Method", result, A, y)
	# Visualization
	graph(formula_lse, formula_newton, x, y)
	