import math

def IsSignificant(p1, p2, n, alpha):

	'''
	This function takes in the arguments p1,p2,n,alpha
	where:
		p1 and p2 are the accuracies of the models
		n is the number of samples
		and alpha is the significance level

	It outputs a boolean 1 (if significant) and 0 (if not significant)
	as well the Z statistic
	'''
	# Swaping accuracies in case the order is swapped
	if p1 > p2:
		temp = p2
		p2 = p1
		p1 = temp

	# Choosing za based off alpha (change to accomodate any alpha?)
	if alpha == 0.1:
		za = 1.28
	elif alpha == 0.05:
		za = 1.645
	elif alpha == 0.025:
		za = 1.96
	elif alpha == 0.01:
		za = 2.576
	else:
		print('Please select from the following alpha values (0.1, 0.05, 0.025).')
		return -1

	# Turning accuracy into percentage
	p1 = p1/100
	p2 = p2/100

	# Creating variales / constants
	x1 = p1*n
	x2 = p2*n
	p_hat = (x1 + x2) / (2*n)

	# Determining Z to compare to - za
	Z = (p1 - p2) / math.sqrt(2*p_hat*(1 - p_hat) / n)

	# Comparing to see if it is significant
	if Z < - za:
		return 1, Z
	else:
		return 0, Z

def main():

	# Baseline 1 and 2
	p1 = 84.24
	p2 = 85.21

	# # Baseline 2 and 3(final)
	# p1 = 81.96
	# p2 = 84.24

	n = 2373  # n is 791 * 3
	alpha = 0.01

	ans, z = IsSignificant(p1,p2,n,alpha)
	print(ans, z)

if __name__ == '__main__':
	main()
