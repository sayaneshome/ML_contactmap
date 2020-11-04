import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras import optimizers
import numpy as np
from sklearn.metrics import confusion_matrix
import time

#Python package keras-metrics could be useful for this (I'm the package's author).
import tensorflow.keras.metrics
#import keras
#import keras_metrics

# creating initial dataframe
Res_pair = ('GLU-LEU','LEU-THR','LEU-MET','GLU-THR','GLU-MET','MET-THR','GLU-SER','GLU-LYS','ALA-GLU','LYS-SER','ALA-SER','ALA-LYS','GLN-GLY','GLN-HIS','ALA-GLN','GLY-HIS','ALA-GLY','ALA-HIS','ASP-GLY','ASP-GLN','GLY-GLY','ALA-ARG','ARG-GLY','ALA-ALA','ILE-LEU','ILE-VAL','ILE-ILE','LEU-VAL','LEU-LYS','LYS-THR','ALA-THR','GLU-GLY')
data = pd.read_csv('Step1_output.csv')
data = data.sample(frac=1).reset_index(drop=True)
data1 = pd.DataFrame(data, columns=['Res_pair'])

# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
data1['Res_pair_ID'] = labelencoder.fit_transform(data1['Res_pair'])
data['Res_pair'] = data1['Res_pair_ID']
data = data.to_numpy()
train_X = data[0:data.shape[0],0:12]
train_y = data[0:data.shape[0],12:data.shape[1]]
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))

def print_statistics(model, x, y):
    predictions = model.predict(x)
    list = []
    for row in predictions:
        list.append(np.argmax(row))
    m = confusion_matrix(y, list)
    sum = 0
    print("Class Accuracies:")
    for i in range(10):
        sum += m[i][i]
        print("Class ", i, ": ", round(m[i][i]/np.sum(m[i])*100, 4))
    print("Overall Accuracy: ", round(sum/np.sum(m), 4)*100)
    print("Confusion Matrix:\n", m)

import random
momentum_rate = random.random()
learning_rate = random.uniform(0,0.2)
neurons = random.randint(10,50)
def precision(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def convolutional_neural_network(x, y):
    print("Hyper-parameter values:\n")
    print('Momentum Rate =',momentum_rate,'\n')
    print('learning rate =',learning_rate,'\n')
    print('Number of neurons =',neurons,'\n')

    startTime = time.clock()
    model = Sequential()
    model.add(Conv1D(filters=64,input_shape=train_X.shape[1:],activation='relu',kernel_size = 3))
    model.add(Flatten())
    model.add(Dense(neurons,activation='relu')) # first hidden layer
    model.add(Dense(neurons, activation='relu')) # second hidden layer
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(neurons, activation='relu')) 
    model.add(Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=momentum_rate, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy',tensorflow.keras.metrics.Precision()])
    history = model.fit(train_X, train_y, validation_split=0.2, epochs=10, batch_size=100)
    endTime = time.clock()
    print("Time = ", endTime-startTime)
    #print("\nTraining Data Statistics:\n")
    #print("CNN Model with Relu Hidden Units and Cross-Entropy Error Function:")
    #print_statistics(model, x, y)


for k in range(100):
    momentum_rate = random.random()
    learning_rate = random.uniform(0,0.2)
    neurons = random.randint(10,50)
    print(convolutional_neural_network(train_X, train_y))




#print("\nTest Data Statistics:\n")

#print("CNN Model with Relu Hidden Units and Cross-Entropy Error Function:")
#convolutional_neural_network(train_X, train)



