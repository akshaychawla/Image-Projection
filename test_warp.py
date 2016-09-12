import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimage
import warp 
import time

def main():
	'''Test the warp function in module warp.py '''

	# load images
	img_dest = mpimage.imread('./images/image2.JPG')
	img_logo = mpimage.imread('./images/google_logo.jpg')

	# load points
	points_image = np.load('./images/points_image.npy')
	points_logo = np.load('./images/points_logo.npy')

	# find homography
	H = warp.estimate_homography(points_logo, points_image)

	# warping
	video_copy = warp.inverse_warp(img_logo, img_dest, H, points_image)

	# show the output
	plt.subplot(131)
	plt.imshow(img_dest); plt.axis('off')
	plt.title('Destination')
	plt.subplot(132)
	plt.imshow(img_logo); plt.axis('off')
	plt.title('Source')
	plt.subplot(133)
	plt.imshow(video_copy); plt.axis('off')
	plt.title('Projected source onto Destination')
	plt.show()

if __name__ == '__main__':
	main()