import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np 
import pickle

#print(tf.__version__)  #displays the version of tensorflow

mnist = tf.keras.datasets.mnist #28x28 images of hand-written datasets 0-9

(x_train,y_train), (x_test, y_test) = mnist.load_data() #splitting the data into training and testing sets.

'''
data varies from [0-225] in an array. 
Actually gives us the pixel data
Also, image digit might seem a bit darker.
'''
# print(x_train[0])

'''
Changes the scale of the digit. Provides the pixel data between [0 - 1]
makes the image digit bit lighter.
'''
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# plt.imshow(x_train[2], cmap = plt.cm.binary) #convert the digit image into grayscale
# plt.show()

#Building Model
model = tf.keras.models.Sequential() #Feed Forward

'''
Since the image as dataset is 28x28 multidimensional array. The best way to feed it 
inside the model is to flatten it first. This is also our first input layer.
'''
model.add(tf.keras.layers.Flatten()) 

'''
2 hidden layers with 128 neurons on each and the activation function as rectified linear unit(relu)
'''
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) 
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

'''
Output layer with 10 classfications[0-9]. 
softmax as activation for the probability distribution i.e., the occurance of all the possible outcomes
'''
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) 

model.compile(optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 3)

val_loss, val_acc = model.evaluate(x_test, y_test) #checking overfitted data
#print(val_loss, val_acc)
predictions = model.predict([x_test])
print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()







