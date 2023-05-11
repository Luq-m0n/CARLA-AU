import glob
import os
import typing
from tensorflow import keras
import tqdm
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

image_shape = (640,380,3)

def parse_tfr_element(element):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'raw_image': tf.io.FixedLenFeature([], tf.string),
        'depth': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    height = content['height']
    width = content['width']
    depth = content['depth']
    label = content['label']
    raw_image = content['raw_image']

    #get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(raw_image, out_type=tf.uint8)
    feature = tf.reshape(feature, shape=[height, width, depth])
    return (feature, label)
    
def get_dataset(tfr_dir: str = "/content/", pattern: str = "*images.tfrecords"):
    files = glob.glob(os.path.join(tfr_dir, pattern), recursive=False)
    print(files)

    #create the dataset
    dataset = tf.data.TFRecordDataset(files)

    #pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset
dataset = get_dataset("train.record")
dataset = dataset.batch(32)
dataset = dataset.map(lambda x, y:(tf.cast(x, tf.float32)/255.0, y))
def get_model(input_shape):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=input_shape,strides=2))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

    return model
 
model = get_model(image_shape)
model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])


model.fit(dataset, epochs=5)




