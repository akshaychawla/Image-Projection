import numpy as np 
import pprint

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


