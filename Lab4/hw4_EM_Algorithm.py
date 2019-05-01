import numpy as np
import numba as nb

@nb.jit
def open_file():
	data_type = np.dtype("int32").newbyteorder('>')
	
	data = np.fromfile("../Lab2/train-images-idx3-ubyte", dtype = "ubyte")
	X = data[4 * data_type.itemsize:].astype("float64").reshape(60000, 28 * 28).transpose()
	X = np.divide(X, 128).astype("int")

	labels = np.fromfile("../Lab2/train-labels-idx1-ubyte",dtype = "ubyte").astype("int")
	labels = labels[2 * data_type.itemsize : ].reshape(60000)
	return X, labels

@nb.jit
def initial_parameters():
	PI = np.full((10, 1), 0.1, dtype=np.float64)
	MU = np.random.rand(28 * 28, 10).astype(np.float64)
	MU_prev = np.zeros((28 * 28, 10), dtype=np.float64)
	Z = np.full((10, 60000), 0.1, dtype=np.float64)
	return PI, MU, MU_prev, Z

@nb.jit
def E_Step(X, MU, PI, Z):
	for n in range(0, 60000):
		temp = np.zeros(shape=(10), dtype=np.float64)
		for k in range(0, 10):
			temp1 = np.float64(1.0)
			for i in range(0, 28 * 28):
				if X[i][n]:
					temp1 *= MU[i][k]
				else:
					temp1 *= (1 - MU[i][k])
			temp[k] = PI[k][0] * temp1
		temp2 = np.sum(temp)
		if temp2 == 0:
			temp2 = 1
		for k in range(0, 10):
			Z[k][n] = temp[k] / temp2
	return Z

@nb.jit
def M_Step(X, MU, PI, Z):
	N = np.sum(Z, axis=1)
	for j in range(0, 28*28):
		for m in range(0, 10):
			temp = np.dot(X[j], Z[m])
			temp1 = N[m]
			if temp1 == 0:
				temp1 = 1
			MU[j][m] = (temp / temp1)
	for i in range(0, 10):
		PI[i][0] = N[i] / 60000
	return MU, PI

@nb.jit
def condition_check(PI, MU, MU_prev, Z, condition):
	temp = 0
	for i in range(0, 10):
		if PI[i][0] == 0 :
			condition = 0
			temp = 1
			temp1 = MU_prev
			PI, MU, temp2, Z = initial_parameters()
			MU_prev = temp1
			break
	if temp == 0:
		condition += 1
	return PI, MU, MU_prev, Z, condition

@nb.jit
def difference(MU, MU_prev):
	temp = 0
	for i in range(0, 28 * 28):
		for j in range(0, 10):
			temp += abs(MU[i][j] - MU_prev[i][j])
	return temp

@nb.jit
def print_MU(MU):
	MU_new = MU.transpose()
	for i in range(0, 10):
		print("\nclass: ", i)
		for j in range(0, 28 * 28):
			if j % 28 == 0 and j != 0:
				print("")
			if MU_new[i][j] >= 0.5:
				print("1", end=" ")
			else:
				print("0", end=" ")
		print("")

@nb.jit
def decide_label(X, labels, MU, PI):
	table = np.zeros(shape=(10, 10), dtype=np.int)
	relation = np.full((10), -1, dtype=np.int)
	for n in range(0, 60000):
		temp = np.zeros(shape=10, dtype=np.float64)
		for k in range(0, 10):
			temp1 = np.float64(1.0)
			for i in range(0, 28 * 28):
				if X[n][i] == 1:
					temp1 *= MU[i][k]
				else:
					temp1 *= (1 - MU[i][k])
			temp[k] = PI[k][0] * temp1
		table[labels[n]][np.argmax(temp)] += 1
	for i in range(1, 11):
		ind = np.unravel_index(np.argmax(table, axis=None), table.shape)
		relation[ind[0]] = ind[1]
		for j in range(0, 10):
			table[ind[0]][j] = -1 * i
			table[j][ind[1]] = -1 * i
	return relation

@nb.jit
def print_labeled_class(MU, relation):
	MU_new = MU.transpose()
	for i in range(0, 10):
		print("\nlabeled class: ", i)
		label = relation[i]
		for j in range(0, 28 * 28):
			if j % 28 == 0 and j != 0:
				print("")
			if MU_new[label][j] >= 0.5:
				print("1", end=" ")
			else:
				print("0", end=" ")
		print("")

@nb.jit
def print_confusion_matrix(X, labels, MU, PI, relation):
	error = 60000
	confusion_matrix = np.zeros(shape=(10,2,2), dtype=np.int)
	for n in range(0, 60000):
		temp = np.zeros(shape=10, dtype=np.float64)
		for k in range(0, 10):
			temp1 = np.float64(1.0)
			for i in range(0, 28 * 28):
				if X[n][i] == 1:
					temp1 *= MU[i][k]
				else:
					temp1 *= (1 - MU[i][k])
			temp[k] = PI[k][0] * temp1
		predict = np.argmax(temp)
		for i in range (0, 10):
			if relation[i] == predict:
				predict = i
				break
		for k in range(0, 10):
			if labels[n] == k:
				if predict == k:
					confusion_matrix[k][0][0] += 1
				else:
					confusion_matrix[k][0][1] += 1
			else:
				if predict == k:
					confusion_matrix[k][1][0] += 1
				else:
					confusion_matrix[k][1][1] += 1
	
	for i in range(0, 10):
		print("\n---------------------------------------------------------------\n")
		print("Confusion Matrix {}: ".format(i))
		print("\t\tPredict number {}\t Predict not number {}".format(i, i))
		print("Is number {}\t\t{}\t\t\t{}".format(i, confusion_matrix[i][0][0], confusion_matrix[i][0][1]))
		print("Isn't number {}\t\t{}\t\t\t{}\n".format(i, confusion_matrix[i][1][0], confusion_matrix[i][1][1]))
		print("Sensitivity (Successfully predict number {})\t: {}".format(i, confusion_matrix[i][0][0] / (confusion_matrix[i][0][0] + confusion_matrix[i][0][1])))
		print("Specificity (Successfully predict not number {})\t: {}".format(i, confusion_matrix[i][1][1] / (confusion_matrix[i][1][0] + confusion_matrix[i][1][1])))
	
	for i in range(0, 10):
		error -= confusion_matrix[i][0][0]
	return error

if __name__ == "__main__":
	X, labels = open_file() # X = 784 * 60000
	PI, MU, MU_prev, Z = initial_parameters() # PI = 10 * 1, MU = 784 * 10, Z = 10 * 60000
	iteration = 0
	condition = 0
	while(True):
		iteration += 1
		# E-step:
		Z = E_Step(X, MU, PI, Z)

		# M-step:
		MU, PI = M_Step(X, MU, PI, Z)

		PI, MU, MU_prev, Z, condition = condition_check(PI, MU, MU_prev, Z, condition)
		gap = difference(MU, MU_prev)
		if gap < 20 and condition >= 8 and np.sum(PI) > 0.95:
			break
		MU_prev = MU
		print_MU(MU)
		print("No. of Iteration: {}, Difference: {}\n".format(iteration, gap))
		print("---------------------------------------------------------------\n")
	
	print("---------------------------------------------------------------\n")
	relation = decide_label(X.transpose(), labels, MU, PI)
	print_labeled_class(MU, relation)
	error = print_confusion_matrix(X.transpose(), labels, MU, PI, relation)
	print("\nTotal iteration to converge: {}".format(iteration))
	print("Total error rate: {}".format(error/60000.0))
