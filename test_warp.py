import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimage
import warp 
import time

def main():
	'''Test the warp function in module warp.py '''

	# load images
	img_dest = mpimage.imread('./images/image.jpg')
	img_logo = mpimage.imread('./images/logo.png')

	# load points
	points_image = np.load('./images/points_image.npy')
	points_logo = np.load('./images/points_logo.npy')

	# find homography
	H = warp.estimate_homography(points_logo, points_image)
	
	# Warp some test points
	op = np.dot(H, np.array([ [294],[139],[1] ]))
	print op[0]/op[2]
	print op[1]/op[2]

	# warping
	video_copy = warp.inverse_warp(img_logo, img_dest, H, points_image)

	# show the output
	plt.imshow(video_copy)
	plt.show()

if __name__ == '__main__':
	main()