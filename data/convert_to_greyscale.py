import os
import cv2
import numpy 

for direc in os.listdir('/home/mv01/PFLD/data/300W_newest/train_data/imgs'):
	if(direc.endswith('.png')):
		img = cv2.imread('/home/mv01/PFLD/data/300W_newest/train_data/imgs/' + direc)
		img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# arr = numpy.asarray(img)
		# print(arr.shape)
		cv2.imwrite('/home/mv01/PFLD/data/300W_newest/train_grey/imgs/' + direc, img_g)
		