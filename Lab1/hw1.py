import sys
import csv

def read_input(filename):
	x = []
	y = []
	with open(filename) as file:
		read = csv.reader(file, delimiter = ',')
		for row in read:
			x.append(float(row[0]))
			y.append(float(row[1]))
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

#def matrix_mul():

if __name__ == "__main__":
	filename = sys.argv[1]
	polynomial_bases = int(sys.argv[2])
	lse_lambda = float(sys.argv[3])
	x, y = read_input(filename)
	A = matrix_A(x, polynomial_bases)
	A_transpose = matrix_transpose(A)
