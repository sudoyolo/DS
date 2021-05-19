#Importing Required Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

#Opening Dataset
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

#Scaling Imagery Data
X = X/255.0

#Building The Model
model = Sequential()

#Adding Layers to the Model
model.add(Conv2D(128, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

#Activation Layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Compiling Model
model.compile(loss="sparse_categorical_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])

#Fitting the data into the model
model.fit(X, y, batch_size=32, epochs=5, validation_split=0.3)

#Saving the Model
model.save('CNN.model')

