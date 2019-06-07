import matplotlib.pyplot as plt
import numpy as np

def read_input():
    x = np.genfromtxt('mnist_X.csv', delimiter=',', dtype=np.float32)
    label = np.genfromtxt('mnist_label.csv', delimiter=',', dtype=np.int)
    return x, label

def get_feature_vectors(covariance):
    eigen_values, eigen_vectors = np.linalg.eig(covariance)
    idx = eigen_values.argsort()[::-1]
    return eigen_vectors[:,idx][:,:2]

def calculate_lower_dimension(origin_data, feature_vector):
    # feature_vector size: 784 * 2
    return np.matmul(origin_data, feature_vector)

def draw(data, label):
    color = ['red', 'green', 'blue', 'brown', 'purple']
    plt.title('PCA')
    for i in range(0, data.shape[0]):
        plt.scatter(data[i][0], data[i][1], s=4, c=color[label[i]-1])
    plt.show()

if __name__ == "__main__":
    x, label = read_input() # x size: 5000 * 784, label size: 5000 * 1
    covariance = np.cov(x.transpose())
    feature_vectors = get_feature_vectors(covariance) # feature_vector size: 784 * 2
    lower_dimension_data = calculate_lower_dimension(x, feature_vectors) # lower_dimension_data size: 5000 * 2
    draw(lower_dimension_data, label)
