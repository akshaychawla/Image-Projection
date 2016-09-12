import numpy as np 
import pprint
import matplotlib.path as mplPath

def estimate_homography(logo_pts, video_pts):
	'''estimates the H matrix (homography) that maps points from destination to source,
	i.e x_src  = H * x_dest. 
		Input  -> logo_pts  = 4x2 matrix of points
				  video_pts = 4x2 matrix of points
		
		Output -> H = 3x3 projection matrix that maps points from video_pts to logo_pts 
	'''

	# A matrix - memory allocation as list
	A = []

	# populating A
	for i in range(4):
		
		ax = [ -video_pts[i,0], -video_pts[i,1], -1, 0, 0, 0, video_pts[i,0]*logo_pts[i,0], video_pts[i,1]*logo_pts[i,0], logo_pts[i,0]];
		ay = [0, 0, 0, -video_pts[i,0], -video_pts[i,1], -1, video_pts[i,0]*logo_pts[i,1], video_pts[i,1]*logo_pts[i,1], logo_pts[i,1]];
		
		A.append(ax)
		A.append(ay)
	
	A = np.array(A)

	# Calculating h
	u,s,vt = np.linalg.svd(A)
	h = vt.T[:,-1]

	# Reconstructing H
	H = np.vstack((h[0:3], h[3:6], h[6:9]))

	return H 

def inverse_warp(logo, video, H, video_pts):
	'''Warp the logo image onto the video image between the video_pts using H (homography) 
	calculated by the warp.estimate_homography function'''

	# create a copy of the original video image
	#video_copy = np.zeros(video.shape[0:3])
	video_copy = np.copy(video)
	
	# create set of points that have to be mapped from the logo image
	bbPath = mplPath.Path(video_pts)
	valid_loc = []
	for x in range(video.shape[1]):
		for y in range(video.shape[0]):
			if (bbPath.contains_point((x,y))):
				valid_loc.append([x,y])
				

	# Iterate through the list
	for pt in valid_loc:
		
		# Projecting points from video to logo 
		pt_lt = np.dot(H, np.array([[pt[0]], [pt[1]], [1] ]) )
		x_l = round(pt_lt[0]/pt_lt[2])
		y_l = round(pt_lt[1]/pt_lt[2])
		
		# applying safe limits to projected points
		x_l = max(x_l,0)
		x_l = min(x_l,logo.shape[1]-1)
		y_l = max(y_l,0)
		y_l = min(y_l,logo.shape[0]-1)

		# copying over pixel data
		video_copy[pt[1],pt[0]] = logo[y_l, x_l]

	# return the video_copy 
	return video_copy

