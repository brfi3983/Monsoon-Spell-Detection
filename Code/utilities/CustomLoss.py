import numpy as np

def CustomLoss(y_true, y_pred):

	'''
	This function is the loss function that takes in y_pred and y_true as arrays:
	y_pred = [[.3 .5 .2], ...]
	y_true = [[0  1  0], ...]

	and manually calculates the Cross Entropy loss given a certain class weight on class 0 and 2 (extremes)
	and then returns the loss.

	*****note that this is for One-Hot y_true arrays*****
	'''
	rows, colns = y_true.shape[0], y_true.shape[1]
	loss = 0
	for i in range (0,rows):
		y_array = y_true[i,:]
		y_hat_array = y_pred[i,:]

		for j in range (0,colns):

			c = j
			y = y_array[j]
			y_hat = y_hat_array[j]

			# Initialize weight
			loss_weight = 1

			# If the class isn't the second element, it will weight the loss by X (2 in our case)
			if c != 1:
				loss_weight = 2

			# Cross entropy loss function with the weight added from above
			val = y*np.log10(y_hat) + (1 - y)*np.log10(1 - y_hat)
			val = loss_weight*val
			loss += val

	return loss

def main():

	x = np.array([[0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])
	y = np.array([[0, 1, 0],[1, 0, 0]])

	print(x.shape)
	print(y.shape)

	test = CustomLoss(x,y)

	print('loss:', test)

if __name__ == "__main__":
	main()
