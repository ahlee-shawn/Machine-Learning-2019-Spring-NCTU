import numpy as np
import math
import time

def univariate_gaussian_data_generator(mean, standard_deviation):
	temp = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
	return mean + standard_deviation * temp

if __name__ == "__main__":
	print("Mean: ", end = " ")
	mean = float(input())
	print("Variance: ", end = "")
	variance = float(input())
	standard_deviation = math.sqrt(variance)

	print("Data point source function: N(", mean, ",", variance, ")\n")

	n = 0
	prev_mean = 0.0
	prev_variance = 0.0
	mle_mean = 0.0
	mle_variance = 0.0

	while(True):
		n += 1
		new_data = float(univariate_gaussian_data_generator(mean, standard_deviation))
		print("Add data point: ", new_data)
		mle_mean = (mle_mean * (n - 1) + new_data) / n
		mle_variance = mle_variance + (prev_mean ** 2) - (mle_mean ** 2) + (((new_data ** 2) - mle_variance - (prev_mean ** 2))/n)
		print("Mean = ", mle_mean, "Variance = ", mle_variance)
		if(abs(prev_mean - mle_mean) < 0.00001 and abs(prev_variance - mle_variance) < 0.00001):
			break
		prev_mean = mle_mean
		prev_variance = mle_variance