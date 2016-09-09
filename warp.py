import numpy as np 


def estimate_homography(src_pts, dest_pts):
	'''estimates the H matrix (homography) that maps points from destination to source,
	i.e x_src  = H * x_dest. 
		Input  -> src_pts  = 4x2 matrix of points
				  dest_pts = 4x2 matrix of points
		
		Output -> H = 3x3 projection matrix that maps points from dest_pts to src_pts 
	'''

	#convert src and dest from cartesian co-ordinates to homogenous
	src_hom, dest_hom = np.ones((4,3)) , np.ones((4,3))
	src_hom[:,0:2] = src_pts
	dest_hom[:,0:2] = dest_pts


	# A matrix - memory allocation
	A = np.zeros((8,9))

	# populating A
	for i in range(4):
		ax = [ [ -dest_hom[i,0], -dest_hom[i,1], -1, 0, 0, 0, -dest_hom[i,0]*src_hom[i,0], dest_hom[i,1]*src_hom[i,0], src_hom[i,0]  ] ]
		ay = [ [0, 0, 0, -dest_hom[i,0], -dest_hom[i,1], -1, dest_hom[i,0]*src_hom[i,1], dest_hom[i,1]*src_hom[i,1], src_hom[i,1] ] ]
		A[ i+ (i-1), :] = np.array(ax)
		A[ i+ (i-1) + 1,:] = np.array(ay);

	# Calculating h
	u,s,vt = np.linalg.svd(A)
	h = vt.T[:,-1]

	# Reconstructing H
	H = np.vstack((h[0:3], h[3:6], h[6:9]))

	return H 


