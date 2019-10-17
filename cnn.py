import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
import numpy as np
import pickle
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.regularizers import l1, l2
import time

trainX = np.load('trainX.npy')
trainY = np.load('trainY.npy')

print(trainX.shape)
print(trainY.shape)


# When testing many combinations
#convLayers = [1, 2, 3]
#layerSizes = [32, 64, 128]
#denseLayers = [0, 1, 2]

# When tweaking best models
convLayers = [3]
convSizes = [64, 128]
denseLayers = [0, 1, 2]
denseSizes = [64, 128, 512]




#droprates = [0, 0.2]

for convLayer in convLayers:
	for convSize in convSizes:
		for denseLayer in denseLayers:
			for denseSize in denseSizes:
				name = "({}-conv_{}-units)-({}-dense_{}-units)-{}".format(convLayer, convSize, denseLayer, denseSize, int(time.time()))
				print(name)
				
				tensorboard = TensorBoard(log_dir='bestModelslogs/{}'.format(name))
				earlystopping = EarlyStopping(monitor='val_loss', patience=1, mode='auto')

				
				model = Sequential()
				
				model.add(Conv2D(convSize, (3,3), input_shape=trainX.shape[1:]))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2,2)))

				for l in range(convLayer-1):
					model.add(Conv2D(convSize, (3,3)))
					model.add(Activation('relu'))
					model.add(MaxPooling2D(pool_size=(2,2)))

				model.add(Flatten())

				for l in range(denseLayer):
					model.add(Dense(denseSize))
					model.add(Activation('relu'))
					#model.add(Dropout(droprate))

				model.add(Dense(7))
				model.add(Activation('softmax'))


				model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
				model.fit(trainX, trainY, 
					batch_size=32, 
					epochs=10, 
					validation_split=0.1, 
					callbacks=[tensorboard])



# Best models
'''
3 conv 64
2 dense 64
10 epochs: 0.597

3 conv 128
1 dense 128
8 epochs: 0.579

3 conv 64
1 dense 64
8 epochs: 0.565

3 conv 128
0 dense 
10 epochs: 0.563

-------------------
(3-conv_64-units)-(1-dense_512-units) 0.5589 6 epochs 
(3-conv_128-units)-(1-dense_128-units) 0.5697 7 epochs
(3-conv_128-units)-(2-dense_512-units) 0.5570 5 epochs




'''
'''

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape=trainX.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))


model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(trainX, trainY, batch_size=32, epochs=10, validation_split=0.1)
'''