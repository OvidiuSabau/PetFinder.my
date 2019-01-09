from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Activation
from keras.layers import Input
import numpy as np

layers = [100, 100, 200, 100, 50]

fc_input = Input(shape=(670, ), dtype=np.float32)

for layer in layers:
    current_activation = Dense(layer,)(current_activation)
    current_activation = BatchNormalization(axis=0)(current_activation)
    current_activation = Activation(activation='relu')(current_activation)

