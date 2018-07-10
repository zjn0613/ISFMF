# coding:utf-8
import os
import h5py
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz/bin/'
import numpy as np
np.random.seed(3934)

from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Reshape, Flatten, Dropout
from keras.layers import Dense, Embedding, concatenate, Input
from keras.preprocessing.image import img_to_array
from keras.preprocessing import sequence
from keras.utils import plot_model
from keras.callbacks import TensorBoard as tb
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras import backend as K


class DDF_module():
    """docstring for DDF_module"""
    batch_size = 256
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, image_size, dropout_rate, emb_dim, max_len, nb_filters):
        super(DDF_module, self).__init__()

        # cluster = tf.train.ClusterSpec({"ps": "157.16.204.236:2222", "worker": "157.16.204.230:2222"})
        # server = tf.train.Server(cluster,
        #                          job_name="worker",
        #                          task_index=0)
        # with tf.device(tf.train.replica_device_setter(
        #         worker_device="/job:worker/task:%d" % 0,
        #         cluster=cluster)):
        #     K.set_learning_phase(1)
        #     K.manual_variable_initialization(True)

        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200

        filter_lengths = [3, 4, 5]

        # DeepContour
        deepContour_model = Sequential()
        deepContour_model.add(Conv2D(32, 5, strides=(1, 1), activation='relu', padding='same', input_shape=(image_size, image_size, 3), name="conv1"))
        deepContour_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        deepContour_model.add(Conv2D(48, 5, strides=(1, 1), activation='relu', padding='same', name="conv2"))
        deepContour_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        deepContour_model.add(Conv2D(64, 5, strides=(1, 1), activation='relu', padding='same', name="conv3"))
        deepContour_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        deepContour_model.add(Conv2D(128, 5, strides=(1, 1), activation='relu', padding='same', name="conv4"))
        deepContour_model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        deepContour_model.load_weights('C:/Users/Microsoft/Desktop/ConvMF/dl_models/deepContour.h5', by_name=True)

        x1 = deepContour_model.output
        x1 = Flatten()(x1)

        # ConvMF
        doc_input = Input(shape=(max_len,), dtype='int32', name='doc_input')

        '''Embedding Layer'''
        sentence_embeddings = Embedding(output_dim=emb_dim, input_dim=max_features, input_length=max_len, name='sentence_embeddings')(doc_input)

        '''Reshape Layer'''
        reshape = Reshape(target_shape=(max_len, emb_dim, 1), name='reshape')(sentence_embeddings)  # chanels last

        '''Convolution Layer & Max Pooling Layer'''
        flatten_ = []
        for i in filter_lengths:
            model_internal = Sequential()
            model_internal.add(Conv2D(nb_filters, (i, emb_dim), activation="relu",
                                      name='conv2d_' + str(i), input_shape=(self.max_len, emb_dim, 1)))
            model_internal.add(MaxPooling2D(pool_size=(self.max_len - i + 1, 1), name='maxpool2d_' + str(i)))
            model_internal.add(Flatten())
            flatten = model_internal(reshape)
            flatten_.append(flatten)

        '''Fully Connect Layer & Dropout Layer'''
        x2 = concatenate(flatten_, axis=-1)

        # Merge two models
        x = concatenate([x1, x2], axis=-1)
        out = Dense(output_dimesion, activation='tanh', name="FC")(x)
        model = Model(inputs=[deepContour_model.input, doc_input], outputs=out)

        for layer in deepContour_model.layers:
            layer.trainable = False
        # model = multi_gpu_model(model, gpus=2)
        model.compile(loss='mse', optimizer='rmsprop')
        # model.summary()
        # plot_model(model, to_file='model.png', show_shapes="true")
        self.model = model

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    def train(self, X_images, X_doc, V, item_weight, seed):
        X_doc = sequence.pad_sequences(X_doc, maxlen=self.max_len)
        np.random.seed(seed)
        X_images = np.random.permutation(X_images)
        np.random.seed(seed)
        X_doc = np.random.permutation(X_doc)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        history = self.model.fit(x=[X_images, X_doc], y=V,
                                 verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch,
                                 sample_weight=item_weight)

        # cnn_loss_his = history.history['loss']
        # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
        # if cnn_loss_his != cmp_cnn_loss:
        #     self.nb_epoch = 1
        return history

    def get_projection_layer(self, images, documents):
        documents = sequence.pad_sequences(documents, maxlen=self.max_len)
        Y = self.model.predict([images, documents], batch_size=len(documents))
        return Y
