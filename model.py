import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pydot
import tensorflow as tf
import keras
import seaborn as sb
from sklearn.metrics import confusion_matrix

train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

# Check if there is any null value in dataset
df = pd.concat([train,test], ignore_index=True)
df.isnull().sum().sort_values(ascending=True)

X_train = train.drop('label',axis = 1)
y_train = train['label']
X_test = test.drop('label',axis = 1)
y_test = test['label']
X_train.shape

#Each example is of 28*28 grayscale image. Many NN accept the image of fixed size
# so we have to resize the image
x_train_reshape = X_train.values.reshape(-1,28,28)
x_test_reshape = X_test.values.reshape(-1,28,28)
x_train_reshape.shape

# There are 10 labels in data set
col_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

# We can print the name of labels
print(col_names[y_train[0]])

# Perform Normalization
x_train=X_train/255
x_test=X_test/255

# Implementing sequential model from tensorflow Deep Learning Library
model = tf.keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))
model.add(keras.layers.Dense(units=24, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

# Train our model with 30 epoch using fit method
model_history = model.fit(x_train,y_train,epochs = 30)
print(model_history.params)