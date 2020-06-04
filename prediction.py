#Student Name: Pemba Lama
#Student ID: 20300218

import numpy as np
import matplotlib.pyplot as plt 
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

from keras.constraints import maxnorm
from keras.optimizers import SGD

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

def train_test_nn():	#training and testing set for fully connected neural network
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	#normalize
	x_train = x_train/255.0
	x_test = x_test/255.0
	return x_train, x_test, y_train, y_test



def train_test_cnn():	#training and testing set for convolutional neural network
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.reshape(x_train.shape[0], 28, 28 ,1).astype('float32')
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

	x_train = x_train/255.0
	x_test = x_test/255.0


	#####Encoded######
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)

	return x_train, x_test, y_train, y_test


def model1():

	x_train, x_test, y_train, y_test = train_test_nn()
	model = Sequential([
			Flatten(input_shape = (28, 28)),
			Dense(128, activation = "relu"),
			Dense(128, activation = "relu"),
			Dense(10, activation = "softmax")])


	model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])

	print('-----------------------Running Model 1----------------------')
	model.fit(x_train, y_train, epochs = 5)
	val_loss, val_acc = model.evaluate(x_test, y_test, verbose = 0) 
	return val_loss, val_acc, model, x_test


'''Below include the CNN model. 
The first CNN layer takes the input of the image of size 28 x 28, with the filter of 32 and and kernel size of (5,5)
with the activation function as 'relu'.
Another CNN layer is stacked with the same filter size, kernel size and activation function.

Similarly the second CNN layer has the filter of 64 and and kernel size of (3, 3)
with the activation function as 'relu'.
Another CNN layer is stacked with the same filter size, kernel size and activation function.

Output layer with 10 classfications[0-9]. 
softmax as activation for the probability distribution i.e., the occurance of all the possible outcomes

'''
def model2():
	x_train, x_test, y_train, y_test = train_test_cnn()
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(28,28, 1), strides = (1,1), padding='same', activation='relu'))
	model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
	model.add(MaxPooling2D(2,2))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same', strides = (1,1), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(2,2))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation = "relu"))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation = "softmax"))

	model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])

	print('-----------------------Running Model 2----------------------')
	model.fit(x_train, y_train, epochs = 2)
	val_loss, val_acc = model.evaluate(x_test, y_test, verbose = 0) 
	return val_loss, val_acc, model, x_test



def model3():
	x_train, x_test, y_train, y_test = train_test_cnn()

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(28,28, 1), strides = (2,2), padding='same', activation='sigmoid', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(2,2))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same', activation='sigmoid', strides = (2,2), kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(2,2))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation = "sigmoid", kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation = "softmax"))

	lrate = 0.002
	epochs = 20
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.8, decay=decay, nesterov=False)
	model.compile(optimizer = sgd, loss = "categorical_crossentropy", metrics = ['accuracy'])

	print('-----------------------Running Model 3----------------------')
	model.fit(x_train, y_train, epochs=epochs)
	val_loss, val_acc = model.evaluate(x_test, y_test, verbose = 0) 
	return val_loss, val_acc, model, x_test


#####Test Accuracy#######
val_loss, val_acc, model, x_test = model3()	#select models: model1(), model2(), model3(), to run the model
print("Accuracy: ", val_acc)
print("Loss: ", val_loss)

predictions = model.predict([x_test])
test_num = 0 # testing numbers from 0 - 10000
result = np.argmax(predictions[test_num])
print("Predicted: ", result)

