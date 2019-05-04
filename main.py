from datetime import datetime
import os,cv2
#from cv2 import getRotationMatrix2D, warpAffine,getAffineTransform,resize,imread,BORDER_REFLECT
import numpy as np
#KERAS IMPORTS
from keras.applications.vgg16 import VGG16
from keras.callbacks import ProgbarLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Conv2DTranspose, Conv2D, concatenate, Dense, Conv1D, TimeDistributed, LSTM, Flatten
from keras.layers.core import Reshape, Activation, Dropout
from keras.preprocessing.image import *
from keras.optimizers import SGD
from dataloader import BlipDatasetLoader

time_window = 10
video_w = 512
video_h = 384
video_c = 3

audio_l = 1024
audio_c = 2

n_epochs = 5

G =  BlipDatasetLoader(16)
train_generator = G.gen()
validation_generator = G.gen(False)

'''
Model Architecture
using Keras functional API
'''

# RGB CNN
inputs_rgb = Input(shape=(time_window, 384, 512, 3))
# keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
vgg_model_rgb = TimeDistributed(VGG16(weights='imagenet', include_top=False))
conv_model_rgb = vgg_model_rgb(inputs_rgb)
conv_model_rgb = TimeDistributed(Flatten())(conv_model_rgb)
dense_rgb = TimeDistributed(Dense(512, activation='relu'))(conv_model_rgb)
# dropout_rgb = Dropout(0.2)(conv_model_rgb)

# STFT CNN
inputs_stft = Input(shape=(time_window, 1025, 2, 1))
conv_model_stft = TimeDistributed(Conv2D(4, kernel_size=(2,2), strides=(1,1), padding='same', activation='tanh', data_format='channels_last'))(inputs_stft)
conv_model_stft = TimeDistributed(Conv2D(8, kernel_size=(2,2), strides=(1,1), activation='tanh', data_format='channels_last'))(conv_model_stft)
conv_model_stft = TimeDistributed(Flatten())(conv_model_stft)
dense_stft = Dense(64, activation='relu')(conv_model_stft)

# concat 
merged = concatenate([dense_rgb, dense_stft], axis=-1)
dense_merged = TimeDistributed(Dense(256, activation='relu'))(merged)
merged_lstm = LSTM(128, return_sequences=True)
dense_merged = TimeDistributed(Dense(32, activation='relu'))(merged_lstm)
out = TimeDistributed(Dense(32, activation='relu'))(dense_merged)
# out = Activation('softmax')(dense_merged)

# welp! i dont know what i'm doing yay!

model = Model(inputs=[inputs_rgb, inputs_stft], outputs=[out])

print('Compiling Model')

model.compile(optimizer=SGD(lr=0.008, decay=1e-6, momentum=0.9, nesterov=True),
              loss='hinge',
              metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=10, verbose=1, epochs=n_epochs, validation_data=validation_generator, validation_steps=10)

print(model.evaluate_generator(validation_generator, steps=100))

model.save('test_model')