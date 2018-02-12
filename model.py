
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

#Brightness augmentation
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2BGR)
    return image1


# In[9]:

#Horizontal and vertical shift
def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.12
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(320, 160))
    
    return image_tr,steer_ang


# In[10]:

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                path_center_image = 'Data1/IMG/'+batch_sample[0].split('\\')[-1]
                path_left_image = 'Data1/IMG/'+batch_sample[1].split('\\')[-1]
                path_right_image = 'Data1/IMG/'+batch_sample[2].split('\\')[-1]
                center_image = cv2.imread(path_center_image)
                left_image = cv2.imread(path_left_image)
                right_image = cv2.imread(path_right_image)
                if center_image == None:
                    print("Invalid image:" , path_center_image)
                else:
                    # add images and angles to data set
                    center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2BGR)
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR)
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR)
                    images.append(center_image)
                    images.append(left_image)
                    images.append(right_image)
                    steering_center = float(batch_sample[3])
                    # create adjusted steering measurements for the side camera images
                    correction = 0.12
                    # 1.0/25 * 3 - the car has a steering angle of -25 to 25 in the simulation
                    # and it is normalized in the driving_log.csv file from -1 to 1,
                    # so 3 angles in either direction would be 0.12
                    steering_left = steering_center + correction
                    steering_right = steering_center - correction
                    # angles to data set
                    angles.append(steering_center)
                    angles.append(steering_left)
                    angles.append(steering_right)
            #Make the brightness augmented images
            brightness_augmented_images, brightness_augmented_angles = [], []
            for image, angle in zip(images, angles):
                brightness_augmented_image = augment_brightness_camera_images(image)
                brightness_augmented_images.append(brightness_augmented_image)
                brightness_augmented_angles.append(angle)
            
            #Add the list of brightness_augmented_images and brightness_augmented_angles to the list of images and angles
            images.extend(brightness_augmented_images)
            angles.extend(brightness_augmented_angles)
            
            #Make the shifted images
            shifted_images, shifted_angles = [],[]
            for image, angle in zip(images, angles):
                shifted_images.append(image)
                shifted_angles.append(angle)
                shifted_image, shifted_angle = trans_image(image, angle, 100)
                shifted_images.append(shifted_image)
                shifted_angles.append(shifted_angle)

            #Make the flipped images and add them to the training data set
            augmented_images, augmented_angles = [],[]
            for image, angle in zip(shifted_images, shifted_angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle * -1.0)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[11]:

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# In[12]:

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# In[13]:

model = Sequential()
#Cropping the images from the top to 70 rows, from the bottom to 25 rows.
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
#Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3)))
##Cropping the images from the top to 70 rows and from the bottom to 25 rows.
#model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(3,160,320)))
#1st convolution layer, Input = 65x320x3, Output = 31x158x24, kernel = 5x5, strides = 2x2, padding=Valid
#Actovate function layer, activate function = RELU
model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, subsample = (2, 2), activation='relu'))
#Dropout layer, keep probability = 0.5
model.add(Dropout(0.5))
#2nd convolution layer, Input = 31x158x24, Output = 14x77x36, kernel = 5x5, strides = 2x2, padding=Valid
#Activate function layer, activate function = RELU
model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample = (2, 2), activation='relu'))
#3rd convolution layer, Input = 14x77x36, Output = 5x37x48, kernel = 5x5, strides = 2x2, padding=Valid
#Activate function layer, activate function = RELU
model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample = (2, 2), activation='relu'))
#4th convolution layer, Input = 5x37x48, Output = 3x35x64, kernel = 3x3, strides = 1x1, padding=Valid
#Activate function layer, activate function = RELU
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu'))
#5th convolution layer, Input = 3x35x64, Output = 1x33x64, kernel = 3x3, strides = 1x1, padding=Valid
#Activate function layer, activate function = RELU
model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu'))
#Flatten, Input = 1x33x64, Output = 2112
model.add(Flatten())
#1st fully connected layer, Input = 2112, Output = 100
model.add(Dense(100))
#2nd fully connected layer, Input = 100, Output = 50
model.add(Dense(50))
#3rd fully connected layer, Input = 50, Output = 1
model.add(Dense(1))


# In[14]:

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)


# In[15]:

model.save('model.h5')


# In[ ]:



