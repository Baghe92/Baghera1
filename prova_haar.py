# -*- coding: utf-8 -*-
"""final_results_gender_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/MITESHPUTHRANNEU/Speech-Emotion-Analyzer/blob/master/final_results_gender_test.ipynb


## Importing the required libraries
"""


import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt, IPython.display as ipd

import cv2

import tensorflow as tf

import os
import pandas as pd 
import scipy.io.wavfile
import pywt
import sys
import json
import random


import tqdm


from tensorflow import keras

from matplotlib.pyplot import specgram
from matplotlib import cm

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import models
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Concatenate
from collections import Counter

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D,Input

from tensorflow.keras.layers import LeakyReLU

#rom tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop


from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score




mylist= os.listdir('RawData/')

data, sampling_rate = librosa.load('RawData/01-01-01-01-01-01-01.wav')

print("\n data", data.shape)


plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)
plt.show()


sr,x = scipy.io.wavfile.read('RawData/01-01-01-01-01-01-01.wav')


df = pd.DataFrame(columns=['feature'])
feeling_list = []
bookmark = 0

hop_length = 441
n_fft = 512


for index,y in enumerate(mylist):
	if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and                mylist[index][:1]!='n' and mylist[index][:1]!='d':
           X, sample_rate = librosa.load('RawData/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
           
           # STFT
           #D = librosa.stft(X, n_fft=n_fft, hop_length=hop_length)
           #S = librosa.amplitude_to_db(abs(D))
           #mel = librosa.feature.melspectrogram(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
           #mel_DB = librosa.power_to_db(mel, ref=np.max)

          
           #if S.shape != (257, 251) :
             # print("Audio file: ", y)
              #input('')
             
           #if S.shape != (257, 251):
             #  continue
           for i in range(0,9):
               (X, coeff_d) = pywt.dwt(X, 'haar')
               array = X
           
           if array.shape == (207,) or array.shape == (209,) or array.shape == (210,) or array.shape == (212,):   
             
               continue
           feature = array
           df.loc[bookmark] = [feature]
           bookmark=bookmark+1
           item = mylist[index]        
           if item[6:-16]=='02':
               feeling_list.append('calm')
           elif item[6:-16]=='03':
               feeling_list.append('happy')
           elif item[6:-16]=='04':
               feeling_list.append('sad')
           elif item[6:-16]=='05':
               feeling_list.append('angry')
           elif item[6:-16]=='06':
               feeling_list.append('fearful')
               
           elif item[:1]=='a':
               feeling_list.append('male_angry')
           elif item[:1]=='f':
               feeling_list.append('male_fearful')
           elif item[:1]=='h':
               feeling_list.append('male_happy')
           #elif item[:1]=='n':
           #feeling_list.append('neutral')
           elif item[:2]=='sa':
               feeling_list.append('male_sad')

               
               
labels = pd.DataFrame(feeling_list)

print(df.shape,labels.shape)


[print("{}: {}".format(i, el.shape)) for i, el in enumerate(df['feature'])]

print('\n')

print(set([el.shape for el in df['feature']]))


x = np.array(df.feature.tolist())
print("\n x.shape.features ",x.shape)

y=labels.to_numpy().ravel()
print("\n y.shape",y.shape)


counter = Counter(y)
print("\n counter",counter)


skf = StratifiedKFold(n_splits = 10)
skf.get_n_splits(x, y)

print(skf)

StratifiedKFold(n_splits = 10, random_state=None, shuffle=False)

for train_index, test_index in skf.split(x, y):
	print("TRAIN:", train_index, "TEST:", test_index)
	x_train, x_test = x[train_index], x[test_index]     
	y_train, y_test = y[train_index], y[test_index]


print('x_train', x_train.shape)
print('x_test' , x_test.shape)
print('y_train', y_train.shape)
print('y_test' , y_test.shape)


lb = LabelEncoder()

y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))

print("\n new y_train",y_train.shape)

x_traincnn =np.expand_dims(x_train, axis=2)
x_testcnn= np.expand_dims(x_test, axis=2)

print("\n new x_train",x_traincnn.shape)


#num_rows     = 224
#num_channels = 1


#x_train = x_traincnn.reshape(x_train.shape[0],num_rows,num_channels)
#x_test  = x_testcnn.reshape(x_test.shape[0],num_rows,num_channels)


#print("\n new x_ train",x_train.shape)



num_labels = y_train.shape[1]

leaky_relu_alpha = 0.1


# Construct model 
model = Sequential()
model.add(Conv1D(32, kernel_size = 5, activation='relu', input_shape=(216,1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2))

model.add(Conv1D(64, 5 , activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(64, 5 , activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2))

model.add(Conv1D(128, 5 , activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(128, 5 , activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size = 2))

model.add(Flatten())
#model.add(Dense(1024, activation='relu'))
model.add(Dense(num_labels, activation='softmax'))

#rms = RMSprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'],optimizer=keras.optimizers.Adam())

# Display model architecture summary 
print(model)


csv_logger = CSVLogger('log.csv', append=True, separator=';')
history = model.fit(x_traincnn, y_train, batch_size=32, epochs=1000, validation_data=(x_testcnn,y_test),callbacks=[csv_logger])


train_score = model.evaluate(x_traincnn, y_train, verbose=0)
print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))


test_score = model.evaluate(x_testcnn, y_test, verbose=0)
print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))







