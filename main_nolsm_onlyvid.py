#%%
from datetime import datetime
import os,cv2
#from cv2 import getRotationMatrix2D, warpAffine,getAffineTransform,resize,imread,BORDER_REFLECT
import numpy as np
#KERAS IMPORTS
from keras.applications.vgg16 import VGG16
from keras.callbacks import ProgbarLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Conv2DTranspose, Conv2D, concatenate, Dense, Conv1D, TimeDistributed, LSTM, Flatten, Bidirectional, BatchNormalization
from keras.layers.core import Reshape, Activation, Dropout
from keras.preprocessing.image import *
from keras.optimizers import SGD
from dataloader import BlipDatasetLoader
from keras.backend import expand_dims, shape
from keras.utils import plot_model
import matplotlib.pyplot as plt

#%%
time_window = 2
video_w = 512
video_h = 384
video_c = 3

audio_l = 1024
audio_c = 2

n_epochs = 5

G =  BlipDatasetLoader(16, frames=time_window, only_vid=True)
train_generator = G.gen(no_timesteps=True)
test_generator = G.gen(False, no_timesteps=True)
validation_generator = G.gen(False, True, no_timesteps=True)

#%%
'''
Model Architecture
using Keras functional API
'''
# first input model

input_rgb = Input(shape=(384, 512, 3))

input_rgb_norm = BatchNormalization(axis=-1)(input_rgb)
conv11a = Conv2D(32, kernel_size=5, padding='same', name="conv11a")(input_rgb)
conv11a = Activation('relu')(conv11a)
conv11b = Conv2D(32, kernel_size=5, padding='same', strides=(2,2), name="conv11b")(conv11a)
conv11b_b = BatchNormalization()(conv11b)
conv11b_a = Activation('relu')(conv11b_b)
conv11b_d = Dropout(0.2)(conv11b_a)
pool11 = MaxPooling2D(pool_size=(3, 3), name="pool11")(conv11b_d)
conv12a = Conv2D(64, kernel_size=3, padding='same', name="conv12a")(pool11)
conv12a = Activation('relu')(conv12a)
conv12b = Conv2D(64, kernel_size=3, padding='same', strides=(2,2), name="conv12b")(conv12a)
conv12b_b = BatchNormalization()(conv12b)
conv12b_a = Activation('relu')(conv12b_b)
conv12b_d = Dropout(0.2)(conv12b_a)
pool12 = MaxPooling2D(pool_size=(2, 2), name="pool12")(conv12b_d)
conv13a = Conv2D(128, kernel_size=3, padding='same', name="conv13a")(pool12)
conv13a = Activation('relu')(conv13a)
conv13b = Conv2D(128, kernel_size=3, padding='same', strides=(2,2), name="conv13b")(conv13a)
conv13b_b = BatchNormalization()(conv13b)
conv13b_a = Activation('relu')(conv13b_b)
conv13b_d = Dropout(0.2)(conv13b_a)
pool13 = MaxPooling2D(pool_size=(2, 2), name="pool13")(conv12b_d)
conv14a = Conv2D(256, kernel_size=3, padding='same', name="conv14a")(pool13)
conv14a = Activation('relu')(conv14a)
conv14b = Conv2D(256, kernel_size=3, padding='same', strides=(2,2), name="conv14b")(conv14a)
conv14b_b = BatchNormalization()(conv14b)
conv14b_a = Activation('relu')(conv14b_b)
conv14b_d = Dropout(0.2)(conv14b_a)
pool14 = MaxPooling2D(pool_size=(2, 2), name="pool14")(conv12b_d)
conv15a = Conv2D(256, kernel_size=3, padding='same', name="conv15a")(pool14)
conv15a = Activation('relu')(conv15a)
conv15b = Conv2D(256, kernel_size=3, padding='same', strides=(2,2), name="conv15b")(conv15a)
conv15b_b = BatchNormalization()(conv15b)
conv15b_a = Activation('relu')(conv15b_b)
conv15b_d = Dropout(0.2)(conv15b_a)
pool15 = MaxPooling2D(pool_size=(2, 2), name="pool15")(conv12b_d)
conv16a = Conv2D(256, kernel_size=3, padding='same', name="conv16a")(pool15)
conv16a = Activation('relu')(conv16a)
conv16b = Conv2D(256, kernel_size=3, padding='same', strides=(2,2), name="conv16b")(conv16a)
conv16b = Activation('relu')(conv16b)
pool16 = MaxPooling2D(pool_size=(2, 2), name="pool16")(conv12b)
conv17a = Conv2D(256, kernel_size=3, padding='same', name="conv17a")(pool16)
conv17a = Activation('relu')(conv17a)
conv17b = Conv2D(256, kernel_size=3, padding='same', strides=(2,2), name="conv17b")(conv17a)
conv17b = Activation('relu')(conv17b)
pool17 = MaxPooling2D(pool_size=(2, 2), name="pool17")(conv12b)
flat1 = Flatten()(pool17)

# conv11 = Conv2D(16, kernel_size=4, padding='valid', activation='relu', name="conv11" ) (input_rgb)
# pool11 = MaxPooling2D(pool_size=(2, 2), name="pool11" ) (conv11)
# conv12 = Conv2D(32, kernel_size=4, padding='valid', activation='relu', name="conv12" ) (pool11)
# pool12 = MaxPooling2D(pool_size=(2, 2), name="pool12" ) (conv12)
# flat1 = Flatten( name="flat1" ) (pool12)

# second input model
# input_stft = Input(shape=(25, 41, 2))
'''
    conv21a = Conv2D(16, kernel_size=3, padding='same', name="conv21a")(input_stft)
    conv21a = Activation('relu')(conv21a)
    conv21b = Conv2D(16, kernel_size=3, padding='same', name="conv21b")(conv21a)
    conv21b_b = BatchNormalization()(conv21b)
    conv21b_a = Activation('relu')(conv21b_b)
    conv21b_d = Dropout(0.2)(conv21b_a)
    pool21 = MaxPooling2D(pool_size=(2, 2), name="pool21")(conv21b_d)
    conv22a = Conv2D(32, kernel_size=3, padding='same', name="conv22a")(pool21)
    conv22a = Activation('relu')(conv22a)
    conv22b = Conv2D(32, kernel_size=3, padding='same', name="conv22b")(conv22a)
    conv22b_b = BatchNormalization()(conv22b)
    conv22b_a = Activation('relu')(conv22b_b)
    conv22b_d = Dropout(0.2)(conv22b_a)
    pool22 = MaxPooling2D(pool_size=(2, 2), name="pool22")(conv22b_d)
    conv23a = Conv2D(32, kernel_size=3, padding='same', name="conv23a")(pool22)
    conv23a = Activation('relu')(conv23a)
    conv23b = Conv2D(32, kernel_size=3, padding='same', name="conv23b")(conv23a)
    conv23b_b = BatchNormalization()(conv23b)
    conv23b_a = Activation('relu')(conv23b_b)
    conv23b_d = Dropout(0.2)(conv23b_a)
    pool23 = MaxPooling2D(pool_size=(2, 2), name="pool23")(conv23b_d)
'''

# conv21 = Conv2D(16, kernel_size=4, activation='relu', name="conv21" )(input_stft)
# pool21 = MaxPooling2D(pool_size=(2, 2), name="pool21" )(conv21)
# conv22 = Conv2D(32, kernel_size=4, activation='relu', name="conv22" )(pool21)
# pool22 = MaxPooling2D(pool_size=(2, 2), name="pool22" )(conv22)
# flat2 = Flatten( name="flat2" )(pool22)

# merge input models
# merge = concatenate([flat1, flat2])

# lstm1 = LSTM(32, return_sequences=True)(merge)
# flatten_lstm = Flatten() (lstm1)

hidden1 = Dense(16, activation='relu') (flat1)
hidden2 = Dense(16, activation='relu') (hidden1)
output = Dense(2, activation='softmax') (hidden2)

model = Model(inputs=input_rgb, outputs=output)

# summarize layers
print(model.summary())
# exit()

model.compile(optimizer=SGD(lr=0.008, decay=1e-6, momentum=0.9, nesterov=True),
              loss='mean_squared_error',
              metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=100, verbose=1, epochs=n_epochs, validation_data=validation_generator, validation_steps=10, use_multiprocessing=True)

print(model.evaluate_generator(test_generator, steps=100))

plot_model(model, to_file='model.png')

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("accuracy_nolstm_onlyvid.png")

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("loss_nolstm_onlyvid.png")

# model.save('final_model')