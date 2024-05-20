# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:55:16 2024

@author: Axel Morain
https://www.linkedin.com/in/axel-morain/


-------------------------------------------------------------------------------
MULTY-CLASS CLASSIFICATION CONVOLUTIONAL NEURAL NETWORK for OPTICAL
CHARACTER RECONGNITION
-------------------------------------------------------------------------------

The goal of this project was to classify images of typed character of different 
font. They are a total of 36 classes, 0 to 9 and the 26 letters of the alphabet. 

The database was acquired from Kaggle
(https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset). 



The project follows the following framework: 

    1. Importing Images 

    2. Data Exploration and Pre-Processing 

    3. Model Building and Testing 


-------------------------------------------------------------------------------
Things I learned and would do differently for my next projects: 

    1.To create the labels, use the tf/keras on hot encoding function instead of the 
sklearn one. The sparse matrix will then be compatible with the tf/keras tools.
In this project, the matrix was small enough to be used as a regular matrix, 
but for larger projects, I need to keep this in mind.  

    2. When training the model, in the ‘fit’ method, having ‘shuffle = True’ 
does not replace shuffling the data beforehand. Reading the documentation 
again, I do not know why it did not work, but anyway... The model had a very 
hard time learning. It’s validation accuracy stagnated at 0.074. A lot of time
and effort was spent on hyper-parameter tuning and model architecture but in
vain. Finally, the shuffle function from sklearn was applied allowing the 
model to learn and reach validation accuracies above 0.90.  

-------------------------------------------------------------------------------





"""


import numpy as np
import pandas as pd
import os#
import glob#
import matplotlib.pyplot as plt
from PIL import Image
import skimage as ski
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import scipy

def Display_Image(image, title = 'An Image', cmap = 'gray'):
    plt.imshow(image, cmap = cmap)
    plt.title(title)
    plt.show()
    plt.clf()

'''
-------------------------------------------------------------------------------
Importing pictures ------------------------------------------------------------
-------------------------------------------------------------------------------
'''
# Importing the test data
print(os.getcwd())
list_testing_data = glob.glob( r'archive (6)\data\testing_data\*\*.png')

test_data_all_images_raw = np.array([np.array(Image.open(fname))\
                             for fname in list_testing_data])
test_data_all_labels = np.array([list_testing_data[i][30]\
                        for i in range(len(list_testing_data))])

test_data_all_images_raw[0].shape
test_data_all_labels.shape

# importing the training data
list_training_data = glob.glob(r'archive (6)\data\training_data\*\*.png')

# This takes a few minutes to run
train_data_all_images_raw = np.array([np.array(Image.open(fname))\
                             for fname in list_training_data])
train_data_all_labels = np.array([list_training_data[i][31]\
                        for i in range(len(list_training_data))])

train_data_all_images_raw[0].shape
train_data_all_labels.shape

del list_testing_data, list_training_data

'''
-------------------------------------------------------------------------------
Data Exploration and Preprocessing --------------------------------------------
-------------------------------------------------------------------------------
'''

# From manual data exploration, I found that not all images have the same 
# dimentions
# Let's have a closer look


train_im_dim_x = [train_data_all_images_raw[i].shape[1] \
                  for i in range(train_data_all_images_raw.shape[0]) ]

train_im_dim_y = [train_data_all_images_raw[i].shape[0] \
                  for i in range(train_data_all_images_raw.shape[0]) ]

plt.hist(train_im_dim_x, bins = 100, color = 'b')
plt.hist(train_im_dim_y, bins = 100, color = 'r')
plt.title('Distribution of the dimentions of trainning images')
plt.legend(('X axis lenght', 'Y axis lengh'))
plt.show()
plt.clf()

# Okay, so the distribution looks fairly normal. Reshaping the images using 
# the mode of each axis will be good enough.
# What I feared was a bimodal distribution, that would have required more 
# work. What we have is alright ! 
#
#------------------------------------------------------
# I do not think that was a good idea. The models built have a very hard time 
# learning.
# It might be due to the shape of the images so let's turn them into squares.
# Why squares? Why would it even make a difference? I don't really know. I
# Just know that all the pre-trained models availables only take as inputs
# square images and output squares as well.
#
#--------------------------------------------------------
# This change did not lead to an improvement. What really helped was the 
# shuffling of the data as described in the header.
# The change was kept regardless
#


#X_dim_mode = float(scipy.stats.mode(train_im_dim_x)[0])
#Y_dim_mode = float(scipy.stats.mode(train_im_dim_y)[0])

X_dim_mode = 32
Y_dim_mode = 32


test = train_data_all_images_raw[0]
print('test shape: ' + str(test.shape))
Display_Image(test, title = 'before reshaping')
test1 = ski.transform.resize(test, (Y_dim_mode, X_dim_mode))
Display_Image(test1, \
              title = 'After reshaping: {0}x{1}'.format(Y_dim_mode, X_dim_mode))
# It is working !
# Let's apply it to the rest of the data:
test_im = np.array([ski.transform.resize(im, (Y_dim_mode, X_dim_mode))\
           for im in test_data_all_images_raw ])

train_im = np.array([ski.transform.resize(im, (Y_dim_mode, X_dim_mode))\
           for im in train_data_all_images_raw ])

del X_dim_mode, Y_dim_mode, test1

# The images are already on a gray scale, which is convenient. But let's 
# have a look at their range.

for i in range(0, 2000, 200):
    plt.hist(train_im[i, :, :].ravel(), bins = 256)
    plt.title('histogram of train image #' + str(i))
    plt.show()
    plt.clf()
# Looking at the histograms the range of brightness is very high, with a very 
# sharp transition between the bright pixels and the darck ones. This means
# high contrast images, wish is a good news! It is what we want !
# 
# Next denoizing or smoothing functions could be applied but after manualy
# scroling through 80ish images this will not be needed.
#
# A few more things can be done such as tresholding, superpixel
# algorithm, and segmentation.
# Let's start by applying tresholding, feed the data in an algorithm and then
# see if more work needs to be done or not.
#------------------------------------------------
# Thresholding was enough !
#

# Let's apply threshold otsu and see how it works.
# First let's find an few interesting letters to test on
Display_Image(train_im[15000, :, :])
Display_Image(train_im[-9000, :, :])
Display_Image(train_im[1500, :, :])

test1 = train_im[15000, :, :]
test2 = train_im[-9000, :, :]
test3 =  train_im[1500, :, :]

# Apply Threshold local
test11 = test1 >= ski.filters.threshold_otsu(test1)
test21 = test2 >= ski.filters.threshold_otsu(test2)
test31 = test3 >= ski.filters.threshold_otsu(test3)

Display_Image(test11, title = 'test11')
Display_Image(test21, title = 'test21')
Display_Image(test31, title = 'test31')
# This is looking great !! Super sharp !
del test1, test2, test3, test11, test21, test31, i, test

test_im_2 = np.zeros(test_im.shape)
np.may_share_memory(test_im, test_im_2)  
test_im_2 = np.array([im >= ski.filters.threshold_otsu(im)\
             for im in test_im])
    
train_im_2 = np.zeros(train_im.shape)
train_im_2 = np.array([im >= ski.filters.threshold_otsu(im)\
             for im in train_im])

# To be fitted into the model the images have to be in the shape [y, x, 1]
# Let's to some reshaping on both training and testing images
test_im_2 = test_im_2.reshape(1008, 32, 32, 1)
Display_Image(test_im_2[112, :, :, :])

train_im_2 = train_im_2.reshape(20628, 32, 32, 1) 


# Sanity check:
Display_Image(train_im_2[15000, :, :])
Display_Image(train_im_2[-9000, :, :])
Display_Image(train_im_2[1500, :, :])

Display_Image(test_im_2[150, :, :])


train_im = train_im_2.reshape(20628, 32, 32, 1) 
test_im = test_im_2.reshape(1008, 32, 32, 1)

# Everything is looking good on the image side!!
# Let's have a look at the labels
# For this multy class classification let's apply one hot encoding to the labels

train_data_all_labels.shape
print(train_data_all_labels[:20])
train_data_all_labels.reshape(-1, 1).shape # That is the shape we need

train_data_all_labels = train_data_all_labels.reshape(-1, 1)
test_data_all_labels = test_data_all_labels.reshape(-1, 1)

ohe = OneHotEncoder(sparse=False)
train_labels = ohe.fit_transform(train_data_all_labels)
test_labels = ohe.transform(test_data_all_labels)

# This is my first time working with sparce matixes and using One Hot Encoding
# on the label data so let's play arround a little bit
test_labels[:10, :]
temp = ohe.inverse_transform(test_labels[:10, :])
print(temp)
# looks good, let's try an other one
temp = ohe.inverse_transform(test_labels[1007-10:1007, :])
print(temp)

# Before going to the next step let's do some variable cleaning and re-naming
# to fit expected conventions

X_train = train_im
y_train = train_labels

X_test = test_im
y_test = test_labels

del  train_labels, test_labels, test_im\
    ,train_im, train_im_dim_x, train_im_dim_y, temp
del test_data_all_images_raw, test_data_all_labels, train_data_all_images_raw\
    ,train_data_all_labels

# At last, let's suffle the training and test sets !

X_train, y_train = shuffle(X_train, y_train, random_state= 5)
X_test, y_test = shuffle(X_test, y_test, random_state=5)


for i in range(10):
    Display_Image(X_train[i], title= i)
    print(y_train[i])
# This looks right !

del i, test_im_2, train_im_2

'''
# Let's visalise some stuff.
for i in range(5000, 10000, 50):
    Display_Image(X_train[i], title= i)

# Let's visalise some stuff.
for i in range(10000, 15000, 50):
    Display_Image(X_train[i], title= i)

# Let's visalise some stuff.
for i in range(15000, 20000, 50):
    Display_Image(X_train[i], title= i)

# Let's visalise some stuff.
for i in range(20000, X_train.shape[0], 50):
    Display_Image(X_train[i], title= i)
'''


'''
-------------------------------------------------------------------------------
Model Building and Testing ----------------------------------------------------
-------------------------------------------------------------------------------
'''


def Plot_History_model(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_accuracy'])
    #plt.plot(history.history['val_false_negatives_7'])
    plt.legend(['loss', 'val_acc Acc'])
    plt.title('Model3 training loss and validation acc')
    plt.show()
    plt.clf()



import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, LeakyReLU, Softmax, ReLU, MaxPooling2D\
                        ,Dropout
from keras.activations import tanh, hard_sigmoid, relu
from keras import Input 
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.optimizers import SGD, RMSprop
import keras.utils
from tensorflow.keras.optimizers.legacy import Adam
#Verify the GPU instalation of tf
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# It should output the make and name of your GPU. If it does not you are running
# tf on the cpu. On my end, I am running this on my RTX 3070 and it takes 
# a few seconds per epochs.

    

def Model4(dropout_rate):
    '''
    I previously tried 3 vastly different models and this one stood out as 
    the best 
    

    Parameters
    ----------
    dropout_rate: dropout rate for the dropout layer. That one layer can be 
    found downstream of the convulotional layers, and in the dense layers.
    
    Returns
    -------
    model4 : keras.engine.sequential.Sequential
        A model ready to be compiled

    '''
    model4 = Sequential()
    model4.add(Input(shape = (32, 32, 1)))
    model4.add(MaxPooling2D((2, 2)
                            ,padding = 'same'))
    # convolution and max pooling combo 1
    model4.add(Conv2D(filters = 25
                      ,input_shape = (32,32, 1)
                      ,kernel_size = 5
                      ,strides = (1, 1)
                      ,padding = 'same'
                      ,activation = ReLU()
                      ))    

    model4.add(MaxPooling2D((2, 2)
                            ,padding = 'same'))   


    # convolution and max pooling combo 2
    model4.add(Conv2D(filters = 25
                      ,kernel_size = 3
                      ,strides = (1,1)
                      ,padding = 'same'
                      ,activation = ReLU()
                      ))
    model4.add(MaxPooling2D((2, 2)
                            ,padding = 'same'))

    
    # convolution and max pooling combo 3
    model4.add(Conv2D(filters = 25
                      ,input_shape = (30, 28, 1)
                      ,kernel_size = 3
                      ,strides = (1, 1)
                      ,padding = 'same'
                      ,activation = ReLU()
                      ))
    model4.add(MaxPooling2D((2, 2)
                            ,padding = 'same'))

    
    # convolution and max pooling combo 4
    model4.add(Conv2D(filters = 25
                      ,kernel_size = 3
                      ,strides = (1,1)
                      ,padding = 'same'
                      ,activation = ReLU()
                      ))
    model4.add(MaxPooling2D((2, 2)
                            ,padding = 'same'))

    
    # convolution and max pooling combo 5
    model4.add(Conv2D(filters = 25
                      ,kernel_size = 3
                      ,strides = (1,1)
                      ,padding = 'same'
                      ,activation = ReLU()
                      ))
    model4.add(MaxPooling2D((2, 2)
                            ,padding = 'same'))


    # Flattening Layer followed by dense layer
    model4.add(Flatten())
    # 23,000 parameters are been outputted by the flatten layer
    model4.add(Dense(units = 256
                ,activation = ReLU()
                ))
    model4.add(Dropout(rate = dropout_rate))

    model4.add(Dense(units = 64
                ,activation = ReLU()
                ))
    model4.add(Dense(units = 64
                ,activation = ReLU()
                ))

    model4.add(Dense(units = 36
                    ,activation = Softmax()))

    return model4
 



model4 = Model4(dropout_rate = 0.25)
# It is very interesting how incresing the drop-out rate makes the model learn 
# faster. At 25% the model validation accuracy overcome 80% at arround 13 epochs
# Without the dropout rate it takes arround 18 epochs to pass the 80% benchmark.
# If we increase the dropout rate to 30% it then becomes detrimental to the 
# model as it takes more than 15 epochs to surpass 80%

model4.compile(optimizer = SGD()\
               ,loss = CategoricalCrossentropy()\
               ,metrics = ['accuracy'])

model4.summary()

history = model4.fit(x= X_train\
                     ,y = y_train\
  #                   , batch_size = 100\
                     ,epochs = 30\
                     , validation_split = .15\
                     , shuffle = True\
                     )

hh = history.history
print(hh.keys())
print('Validation_acc: {}'.format(hh['val_accuracy']))

plt.scatter(x = list(range(len(hh['val_accuracy'])))\
            ,y = hh['val_accuracy']\
            )
plt.title('Plot of validation accuracy')
plt.show()
plt.clf()

Plot_History_model(history)

#
# Nice ! I am pretty sure this is the best model architecture we can build.
# Let's add a call back, extract the best model possible and feed it our test
# data.


from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stoping = EarlyStopping(monitor = 'val_loss'\
                              ,patience = 10\
                              )
checkpoint_filepath = 'best_model.hdf5'

model_checkpoint = ModelCheckpoint(filepath = checkpoint_filepath\
                                   ,monitor = 'val_accuracy'
                                   ,save_best_only = True\
                                   ,)

history = model4.fit(x= X_train\
                     ,y = y_train\
          #                   , batch_size = 100\
                     ,epochs = 60\
                     , validation_split = .15\
                     , shuffle = True\
                     ,callbacks = [early_stoping, model_checkpoint]
                     )

model4.load_weights(checkpoint_filepath)

#
# Let's compare the 
#

train_metrics = model4.evaluate(x= X_train\
                                ,y = y_train\
                                    )

test_metrics = model4.evaluate(x= X_test\
                               ,y = y_test\
                                   )
print('Test loss: {0}, test accuracy {1}'.\
      format(round(test_metrics[0], 2),\
             round(test_metrics[1], 2)))
# 99% percent accuracy !!!! Let's GOOOOOOOoooooOOOOooOO
#
#
#     !!! PROJECT COMPLETED !!!! 
#
#
#
