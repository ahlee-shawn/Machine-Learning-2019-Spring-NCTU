import sys
import csv

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

if __name__ == "__main__":
	filename = sys.argv[1]
	polynomial_bases = int(sys.argv[2])
	lse_lambda = float(sys.argv[3])
	x, y = read_input(filename)
	A = matrix_A(x, polynomial_bases)
	A_transpose = matrix_transpose(A)
	A_transpose_mul_A = matrix_mul(A_transpose, A)
	lambda_I = I_mul_scalar(lse_lambda, len(A_transpose_mul_A))
	A_transpose_mul_A_add_lambda_I = matrix_add(A_transpose_mul_A, lambda_I)
	A_transpose_mul_b = matrix_mul(A_transpose, y)
