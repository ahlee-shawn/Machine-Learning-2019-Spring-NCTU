import sys
import math
import time

def factorial(x):
	if x > 2:
		return x * factorial(x - 1)
	else:
		return 2

def online_learning():
	filename = sys.argv[1]
	a = int(sys.argv[2])
	b = int(sys.argv[3])
	data = open(filename,'r').read().split('\n')
	for case in range(len(data)):
		line = data[case]
		case = case + 1
		print("case ", case, ": ", line)
		m = 0
		N = len(line)
		for i in range(N):
			if line[i] == '1':
				m += 1
		likelihood = ( factorial(N) / ( factorial(m) * factorial(N - m) ) ) * ((m / N) ** m) * ((1 - m / N) ** (N - m))
		print("Likelihood: ", likelihood)
		print("Beta prior:\ta = ", a, "\tb = ", b)
		a += m
		b += (N - m)
		print("Beta posterior:\ta = ", a, "\tb = ", b)
		print("")

if __name__ == "__main__":
	online_learning()