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


import tensorflow as tf

import os
import pandas as pd
import glob 
import scipy.io.wavfile
import sys
import json
import random

import soundfile
import pickle
import tqdm


from tensorflow import keras

from matplotlib.pyplot import specgram
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input

#from tensorflow.keras.utils import np_utils 
#from keras.layers import Input, Flatten, Dropout, Activation
#from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D


from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization, MaxPooling2D,GlobalAveragePooling2D,Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import LeakyReLU


#from keras.layers import Dense, Conv2D, Flatten,Dropout
#from keras.layers import Dense, Input, Dropout, BatchNormalization, Convolution2D, MaxPooling2D, GlobalMaxPool2D

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from datetime import datetime




mylist= os.listdir('RawData/')


data, sampling_rate = librosa.load('RawData/01-01-01-01-01-01-01.wav')

print(type(data), type(sampling_rate))

print('data.shape',data.shape)
print('sampling_rate',sampling_rate)



plt.figure(figsize=(14, 5))
librosa.display.waveplot(data, sr=sampling_rate)

plt.show()




# stft

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)
sr,x = scipy.io.wavfile.read('RawData/01-01-01-01-01-01-01.wav')

print('sr',sr)

print('x',x.shape)


#X = librosa.stft(data)
#Xdb = librosa.amplitude_to_db(abs(X))

hop_length = 441
n_fft = 512
D = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)

#float(hop_length)/sr # units of seconds
#float(n_fft)/sr  # units of seconds

print('D',D.shape)


print('\n')

S = librosa.amplitude_to_db(abs(D))

print('S',S)
print(S.shape)

plt.figure(figsize=(15, 5))
librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')

plt.show()
print('\n stft.shape')
print(S.shape)


feeling_list = []

features = []
#labels = pd.DataFrame(feeling_list)

labels = pd.DataFrame(columns=['feeling_list'])

df = pd.DataFrame(columns=['feature'])
bookmark = 0

cont = 0
#iterrows
#for index,y in range(len(mylist)):
for index,y in enumerate(mylist):
   
	hop_length = 441
	n_fft = 512
	if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and     		   mylist[index][:1]!='n' and mylist[index][:1]!='d':
           X, sample_rate = librosa.load('RawData/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.6)
           # STFT
           D  = librosa.stft(X, n_fft=n_fft, hop_length=hop_length)
           float(hop_length)/sample_rate # units of seconds
           float(n_fft)/sample_rate  # units of seconds
           
           S = librosa.amplitude_to_db(abs(D))
          
           #if S.shape != (257, 251) :
             # print("Audio file: ", y)
              #input('')
             
           if S.shape != (257, 251):
           	continue
           feature = S
           df.loc[bookmark] = [feature]
           bookmark=bookmark+1
           
           if mylist[6:-16] == '02' and int(mylist[18:-4])%2 == 0:
           	feeling_list.append('female_calm')
           elif mylist[6:-16] == '02' and int(mylist[18:-4])%2 == 1:
           	feeling_list.append('male_calm')
           elif mylist[6:-16] == '03' and int(mylist[18:-4])%2 == 0:
           	feeling_list.append('female_happy')
           elif mylist[6:-16] == '03' and int(mylist[18:-4])%2 == 1:
           	feeling_list.append('male_happy')
           elif mylist[6:-16] == '04' and int(mylist[18:-4])%2 == 0:
           	feeling_list.append('female_sad')
           elif mylist[6:-16] == '04' and int(mylist[18:-4])%2 == 1:
           	feeling_list.append('male_sad')
           elif mylist[6:-16] == '05' and int(mylist[18:-4])%2 == 0:
           	feeling_list.append('female_angry')
           elif mylist[6:-16] == '05' and int(mylist[18:-4])%2 == 1:
           	feeling_list.append('male_angry')
           elif mylist[6:-16] == '06' and int(mylist[18:-4])%2 == 0:
           	feeling_list.append('female_fearful')
           elif mylist[6:-16] == '06' and int(mylist[18:-4])%2 == 0:
           	feeling_list.append('male_fearful')
           elif mylist[:1] == 'a':
           	feeling_list.append('male_angry')
           elif mylist[:1] == 'f':
           	feeling_list.append('male_fearful')
           elif mylist[:1] == 'h':
           	feeling_list.append('male_happy')
           elif mylist[:1] == 'sa':
           	feeling_list.append('male_sad')
           labels.loc[cont] = [feeling_list]
           cont = cont+1
           
           features.append([feature, labels])	
           
 # Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','labels'])          


#labels = pd.DataFrame(feeling_list)


print('df.shape',df.shape)


#print('df.shape',labels.shape)


df.to_csv()

x = df.to_numpy()

x = np.array(df['feature'].tolist())


print(x.shape)

y = np.array(labels['feeling_list'].tolist())


print('y.shape',y.shape)

#lb = LabelEncoder()


#yy = to_categorical(lb.fit_transform(y))



#print('y_train',yy.shape)


#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 42)



#print('x_train', x_train.shape)
#print('x_test' , x_test.shape)
#print('y_train', y_train.shape)
#print('y_test' , y_test.shape)




















'''

x_train, x_test, y_train, y_test = train_test_split(x, yy, test_size=0.20, random_state = 42)



print('x_train', x_train.shape)
print('x_test' , x_test.shape)
print('y_train', y_train.shape)
print('y_test' , y_test.shape)

x_train[0].flatten()

x_traincnn =np.expand_dims(x_train, axis=2)
x_testcnn= np.expand_dims(x_test, axis=2)

print('x_traincnn',x_traincnn.shape)




num_rows     = 257
num_columns  = 258
num_channels = 1

leaky_relu_alpha = 0.1


x_train = x_train.reshape(x_train.shape[0],num_rows,num_columns,num_channels)
x_test  = x_test.reshape(x_test.shape[0],num_rows,num_columns,num_channels)

print('new_train',x_train.shape)

num_labels = yy.shape[1]

'''


