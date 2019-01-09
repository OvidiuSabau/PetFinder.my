from keras.applications import InceptionV3
import keras as k
from utilities import *

def cnn(images):
    model = InceptionV3(include_top = False, weights='imagenet')
    cnn_output = model.predict(images)
    return cnn_output