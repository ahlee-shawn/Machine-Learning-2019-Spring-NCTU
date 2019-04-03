import numpy as np
import math

def univariate_gaussian_data_generator(mean, standard_deviation):
	temp = np.sum(np.random.uniform(0.0, 1.0, 12)) - 6
	return mean + standard_deviation * temp

if __name__ == "__main__":

	# univariate gaussian data generator
	print("Mean: ", end="")
	mean = float(input())
	print("Variance: ", end = "")
	variance = float(input())
	standard_deviation = math.sqrt(variance)
	print(univariate_gaussian_data_generator(mean, standard_deviation))