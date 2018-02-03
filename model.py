
# coding: utf-8

# In[1]:

import os


# In[2]:

import csv


# In[3]:

import cv2


# In[4]:

import numpy as np


# In[5]:

samples =[]
with open('Data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


# In[6]:

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# In[7]:

import sklearn


# In[8]:

#Check whether the shape of the image is correct or not.

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'Data1/IMG/'+batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                if center_image == None:
                    print("Invalid image:" , name)
                else:
                    images.append(center_image)
                    center_angle = float(batch_sample[3])
                    angles.append(center_angle)

                #Make the flipped images and add them to the training data set
            augmented_images, augmented_angles = [],[]
            for center_image, center_angle in zip(images, angles):
                augmented_images.append(center_image)
                augmented_angles.append(center_angle)
                augmented_images.append(cv2.flip(center_image,1))
                augmented_angles.append(center_angle * -1.0)
                
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[9]:

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# In[10]:

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# In[11]:

model = Sequential()
#Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
#Cropping the images from the top to 70 rows and from the bottom to 25 rows.
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))
#1st convolution layer, Input = 65x320x3, Output = 61x316x6, kernel = 5x5, strides = 1x1, padding=Valid
#Actovate function layer, activate function = RELU
model.add(Convolution2D(nb_filter=6, nb_row=5, nb_col=5, activation='relu'))
#1st maxPooling layer, filter size = 2x2, strides = 2, Input = 61x158x6, Output = 31x80x6
model.add(MaxPooling2D((2,2)))
#2nd convolution layer, Input = 31x80x6, Output = 25x76x16, kernel = 5x5, strides = 1x1, padding=Valid
#Actovate function layer, activate function = RELU
model.add(Convolution2D(nb_filter=16, nb_row=5, nb_col=5, activation='relu'))
#2nd maxPooling layer, filter size = 2x2, strides = 2, Input = 25x76x16, Output = 13x38x16
model.add(MaxPooling2D((2,2)))
#Flatten, Input = 13x38x16, Output = 7904
model.add(Flatten())
#1st fully connected layer, Input = 7904, Output = 120
model.add(Dense(120))
#2nd fully connected layer, Input = 120, Output = 84
model.add(Dense(84))
#3rd fully connected layer, Input = 120, Output = 1
model.add(Dense(1))


# In[12]:

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)


# In[13]:

model.save('model.h5')


# In[ ]:



