import matplotlib.pyplot as plt
import numpy as np
import re

def read_pgm(filename, byteorder='>'):
	with open(filename, 'rb') as f:
		buffer_ = f.read()
	header, width, height, maxval = re.search(
		b"(^P5\s(?:\s*#.*[\r\n])*"
		b"(\d+)\s(?:\s*#.*[\r\n])*"
		b"(\d+)\s(?:\s*#.*[\r\n])*"
		b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer_).groups()
	return np.frombuffer(buffer_, dtype='u1' if int(maxval) < 256 else byteorder+'u2', count=int(width)*int(height), offset=len(header))

def read_input():
	data = np.zeros([400, 10304], dtype=np.float32)
	root = './att_faces/'
	target_number = np.random.randint(low=0, high=400, size=10)
	target_number = np.sort(target_number)
	target_filename = []
	target = np.zeros([10, 10304], dtype=np.float32)
	p = 0
	for i in range(1, 41):
		for j in range(1, 11):
			k = (i-1) * 10 + (j-1)
			filename = root + "s" + str(i) + "/" + str(j) + ".pgm"
			temp = read_pgm(filename)
			data[k] += temp
			if p < 10 and target_number[p] == k:
				target[p] += temp
				p += 1
				target_filename.append(filename)
	return data, target, target_filename

def get_feature_vectors(covariance):
	eigen_values, eigen_vectors = np.linalg.eigh(covariance)
	idx = eigen_values.argsort()[::-1]
	return eigen_vectors[:,idx][:,:25]

def draw(reconstructed_data, target_filename):
	for i in range(0, 10):
		plt.clf()
		plt.suptitle(target_filename[i])
		plt.imshow(reconstructed_data[i].reshape(112, 92), plt.cm.gray)
		plt.show()

if __name__ == "__main__":
	data, target, target_filename = read_input() # data size: 400 * 10304, target size: 10 * 10304, target_filename size: 10
	covariance = np.cov(data.transpose()) # covariance size: 10304 * 10304
	feature_vectors = get_feature_vectors(covariance) # feature_vector size: 10304 * 25
	lower_dimension_data = np.matmul(target, feature_vectors)
	reconstructed_data = np.matmul(lower_dimension_data, feature_vectors.transpose())
	draw(reconstructed_data, target_filename)
	