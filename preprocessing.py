import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2 # For image operations
import random



IMG_SIZE = 64

def preProcess(path):
	data = []
	for label, name in enumerate(os.listdir(path)):
		print(name)
		if name == '.DS_Store':
				continue
		for img in os.listdir(path+name):
			try:
				
				imgArray = cv2.imread(path+name + '/' + str(img), cv2.IMREAD_GRAYSCALE)
				newArray = cv2.resize(imgArray, (IMG_SIZE, IMG_SIZE))
				data.append([newArray, label])
				#plt.imshow(imgArray, cmap="gray")
				#plt.show()
			except Exception as e:
				print("Problem reading image.", e)
				pass
			
		
	return data


trainingData = preProcess('/Users/axeloh/Koding/machine_learning/datasets/intel-image-classification/seg_train/')
print(len(trainingData))

random.shuffle(trainingData)
#random.shuffle(testData)

trainX = []
trainY = []

#testX = []
#testY = []

for features, label in trainingData:
	trainX.append(features)
	trainY.append(label)



trainX = np.array(trainX).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#testX = np.array(testX).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


trainX = np.array(trainX)
trainY = np.array(trainY)
#testX = np.array(testX)


trainX = trainX/255.0
#testX = testX/255.0


print(trainX.shape)
print(trainY.shape)


print("Saving data..")

np.save('./trainX.npy', trainX)
np.save('./trainY.npy', trainY)	