import matplotlib.pyplot as plt
import numpy as np

def read_input():
	x = np.genfromtxt('mnist_X.csv', delimiter=',', dtype=np.float32)
	label = np.genfromtxt('mnist_label.csv', delimiter=',', dtype=np.int)
	return x, label

def compute_mean(x, label):
	mean = np.zeros([5, 784], dtype=np.float32)
	for i in range(0, x.shape[0]):
		for j in range(0, x.shape[1]):
			mean[int(i / 1000)][j] += x[i][j]
	for i in range(0, 5):
		for j in range(0, x.shape[1]):
			mean[i][j] /= 1000
	overall_mean = np.mean(x, axis=0)
	return mean, overall_mean

def compute_within_class_scatter_matrix(x, mean):
	within_class = np.zeros([x.shape[1], x.shape[1]], dtype=np.float32)
	for i in range(0, x.shape[0]):
		temp = np.subtract(x[i], mean[int(i / 1000)]).reshape(x.shape[1], 1)
		within_class += np.matmul(temp, temp.transpose())
	return within_class

def compute_between_class_scatter_matrix(mean, overall_mean):
	between_class = np.zeros([x.shape[1], x.shape[1]], dtype=np.float32)
	for i in range(0, 5):
		temp = np.subtract(mean[i], overall_mean).reshape(x.shape[1], 1)
		between_class += np.matmul(temp, temp.transpose())
	between_class *= 1000
	return between_class

def get_feature_vectors(within_class, between_class):
	# since the inverse of matrix within_class does not exist, pseudo inverse is used instead
	eigen_values, eigen_vectors = np.linalg.eig(np.matmul(np.linalg.pinv(within_class), between_class))
	idx = eigen_values.argsort()[::-1]
	return eigen_vectors[:,idx][:,:2]

def draw(data, label):
    color = ['red', 'green', 'blue', 'brown', 'purple']
    plt.title('LDA')
    for i in range(0, data.shape[0]):
        plt.scatter(data[i][0], data[i][1], s=4, c=color[label[i]-1])
    plt.show()

if __name__ == "__main__":
	x, label = read_input() # x size: 5000 * 784, label size: 5000 * 1
	mean, overall_mean = compute_mean(x, label) # mean size: 5 * 784
	within_class = compute_within_class_scatter_matrix(x, mean) # within_class size: 784 * 784
	between_class = compute_between_class_scatter_matrix(mean, overall_mean) # between_class size: 784 * 784
	feature_vectors = get_feature_vectors(within_class, between_class) # feature_vectors size: 784 * 2
	lower_dimension_data = np.matmul(x, feature_vectors)
	draw(lower_dimension_data, label)