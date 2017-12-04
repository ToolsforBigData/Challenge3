from __future__ import division
from os import listdir
from os.path import isfile, join
import mmh3
import imageio # NEED ffmpeg: install using conda: conda install ffmpeg -c conda-forge
from PIL import Image
import numpy as np
from datetime import datetime
import multiprocessing as mp
import imagehash
import operator
from functools import reduce
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import adjusted_rand_index as ARI
import pickle
import warnings

#=============================================
# Pre processing
#=============================================
def pre_process(img):

	#=======CONVERT TO GREYSCALE===============
	img = img.convert('L') #Change to greyscale
	#==========ROTATE AND CROP=================
	width = img.size[0]
	height = img.size[1]
	
	if width < height: # Some videos are vertical but not horizontal
		img = img.rotate(90,expand=1).resize((img.size[1],img.size[0]))
		width = img.size[0]
		height = img.size[1]

	half_the_width = width / 2
	half_the_height = height / 2
	img = img.crop((half_the_width - 226,half_the_height - 127,half_the_width + 226,half_the_height + 127))
	#========HISTOGRAM EQUALIZER==============
	h = img.histogram()
	lut = []
	for b in range(0, len(h), 256):
		# step size
		step = reduce(operator.add, h[b:b+256]) / 255
		# create equalization lookup table
		n = 0
		for i in range(256):
			lut.append(n / step)
			n = n + h[i+b]
	img = img.point(lut)
    
	return img
#============================================

#=============================================
#    Main function for multi processing
#=============================================
def hash_video(video_file_name):
	#-------IMPORTANT PARAMETERS DEFINED------
	buckets = 300 			# Size of the feature vector
	no_frames_to_use = 75   # More frames => better score, slower time
	resolution = 5 # Resolution of the LSH, default 8x8 downsample
	#----------------------------------------

	#----Get video and number of frames-----	
	PATH = '/data/videos/'
	filename = PATH + video_file_name
	vid = imageio.get_reader(filename,  'ffmpeg')
	no_frames = vid.get_length()
	#---------------------------------------
	
	#Compute the gap so we use same amount of frames 
	gap = int(np.ceil(no_frames/no_frames_to_use))
	lst= list([0]*buckets) 

	for img_idx in range(0,no_frames,gap):
		#Convert to image object
		img = Image.fromarray(vid.get_data(img_idx))
		#Preprocess image
		img_processed = pre_process(img)
		#LSH hashing
		hash_string = imagehash.average_hash(img_processed,hash_size=resolution)
		#Feature hashing
		hashed_token = mmh3.hash(str(hash_string)) % buckets
		lst[hashed_token] += 1

	feature_vector = lst

	return (feature_vector,video_file_name[0:-4])


#=============================================
#    CLUSTERING ALGORITHM FUNCTION
#=============================================
def cluster_videos(features,videos):

	#Normalize
	features_standard =  StandardScaler().fit_transform(features)
	#Compute approx. no. clusters
	HOW_MANY_CLUSTERS = int(len(features)/10)
	#Spectral clustering
	spectral = cluster.SpectralClustering(
		n_clusters=HOW_MANY_CLUSTERS+4,
		eigen_solver='arpack', #lobpcg and amg give errors (leading minor of the array is not positive definite)
		assign_labels = 'discretize', #The strategy to use to assign labels in the embedding space. There are two ways to assign labels after the laplacian embedding. k-means can be applied and is a popular choice. But it can also be sensitive to initialization. Discretization is another approach which is less sensitive to random initialization.
		affinity = "nearest_neighbors") #'sigmoid', 'rbf', 'nearest_neighbors'

	# So we don't print out any warnings
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			message="Graph is not fully connected, spectral embedding" +
			" may not work as expected.",
			category=UserWarning)

		spectral.fit(features_standard)
	labels = spectral.labels_
	
	return labels


if __name__ == '__main__':
	#Start timer
	t0 = datetime.now()

	#==========GET ALL FILES=============
	#The path to video files folder
	PATH = '/data/videos/'
	#Find all filenames in folder.
	file_names = [f for f in listdir(PATH) if isfile(join(PATH, f))]
	no_files = len(file_names)
	#====================================
	

	#========MULTIPROCESSING PART==========
	p = mp.Pool(mp.cpu_count()) #Start a multiprocess with 4 processes.
	hash_container = p.map(hash_video,file_names)
	#Turn off multiprocesses
	p.close()
	p.join()
	#======================================

	t1 = datetime.now()

	#========GET RESULTS TO LISTS===========
	#Initializers
	features = []
	names = []
	#Create lists for clustering and printing.
	for item in hash_container:	
		features.append(item[0])
		names.append(item[1])
	#====SAVE TO FILE JUST IN CASE==========
	with open("features.txt", "wb") as fp:   #Pickling
		pickle.dump(features, fp)
	with open("names.txt", "wb") as fb:   #Pickling
		pickle.dump(names, fb)
	#=========================================

	#================CLUSTERING =================
	labels = cluster_videos(features,names)
	#============================================
	
	t2 = datetime.now()

	#===========CHECK ADJ. RAND INDEX =====================

	#Sort the video name list by the clustered labels 
	sidx = np.argsort(labels)
	split_idx = np.flatnonzero(np.diff(np.take(labels,sidx))>0)+1
	out = np.split(np.take(names,sidx,axis=0), split_idx)
	clusters = list(set(L) for L in out)

	#Run Davids true clusters and comp Adj.Rand Index
	score = ARI.rand_index(clusters)

	#=============PRINT SOME NICE STUFF OUT================
	print('Computational time')
	print('Processing: {}\nClustering: {}\nOverall: {}\n'.format(t1-t0,t2-t1,t2-t0))
	print('Adj. Rand Index: {:.5f}'.format(score))
	#=======================================================


