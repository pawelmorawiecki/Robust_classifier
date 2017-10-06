
# coding: utf-8

# In[1]:


import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
import numpy as np
import PIL
from PIL import Image

from skimage import color, exposure, transform
from skimage import io
from skimage import img_as_ubyte

import numpy as np
import hashlib


def cnn_model(IMG_SIZE, NUM_CLASSES):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', 
                            input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                            activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', 
                            activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

def preprocess_img(IMG_SIZE, img):
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE), order=0)

    # roll color axis to axis 0
    #img = np.rollaxis(img,-1)
    
    #convert to byte format (0-255)
    img = img_as_ubyte(img)
    
    return img


def reduce_palette(images):
    
    mask = int('11000000', 2)
    zero = int('00000000', 2)
    
    filter_1 = np.bitwise_and(images[:,:,:,0:1], mask) #leave 2 MSB of the first filter
    filter_2 = np.bitwise_and(images[:,:,:,1:2], zero) #zero the second filter
    filter_3 = np.bitwise_and(images[:,:,:,2:3], zero) #zero the third filter
    
    new_images = np.concatenate((filter_1, filter_2, filter_3), axis=3)
    
    return new_images


def change_pixel_values(IMG_SIZE, image, prefix):
    
    new_image = np.zeros((IMG_SIZE, IMG_SIZE, 3),dtype='uint8')
    for x in range(0,IMG_SIZE):
        for y in range(0,IMG_SIZE):
            # lets concatenate values from all 3 RGB filters
            pixel = image[x][y][0].astype('S10') + image[x][y][1].astype('S10') + image[x][y][2].astype('S10')
            digest = hashlib.sha256(prefix + pixel).hexdigest()
            index = 0 
            for z in range(0,3):
                new_image[x][y][z] = int(digest[index:index+2],16)
                index = index + 2
    return new_image

def get_class(img_path):
    return int(img_path.split('/')[-2])