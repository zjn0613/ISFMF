import os
import numpy as np
np.random.seed(3934)

from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing import sequence
from keras.preprocessing.image import img_to_array
from keras.utils import plot_model
from keras.callbacks import TensorBoard as tb
import tensorflow as tf  # add
from keras.utils.training_utils import multi_gpu_model  # add


class ISF_module():
    """docstring for ISF_module"""
    batch_size = 256
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, image_size):
        super(ISF_module, self).__init__()

        model = Sequential()
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        # model.add(Dropout(0.5))
        model.add(Dense(output_dimesion, activation='tanh', name="FC2"))
        # model = multi_gpu_model(model, gpus=2)
        model.compile(loss='mse', optimizer='rmsprop')
        self.model = model

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def train(self, X_train, V, item_weight, seed):
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        history = self.model.fit(x=X_train, y=V,
                                 verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch,
                                 sample_weight=item_weight)

        # cnn_loss_his = history.history['loss']
        # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
        # if cnn_loss_his != cmp_cnn_loss:
        #     self.nb_epoch = 1
        return history

    def get_projection_layer(self, image):
        Y = self.model.predict(image, batch_size=len(image))
        return Y
