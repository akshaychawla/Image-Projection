import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimage
import warp 


def main():
	'''Test the warp function in module warp.py '''

	# load images
	img_dest = mpimage.imread('./images/image.jpg')
	img_logo = mpimage.imread('./images/logo.png')

	# load points
	points_image = np.load('./images/points_image.npy')
	points_logo = np.load('./images/points_logo.npy')

	# find homography
	H = warp.estimate_homography(points_image, points_logo)

	print H
	op = np.dot(H, np.array([ [989],[399],[1]  ]))
	print op[0,0]/op[2,0]
	print op[1,0]/op[2,0]



if __name__ == '__main__':
	main()