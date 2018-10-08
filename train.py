from __future__ import print_function
import os
import tensorflow as tf; tf.Session()
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"
from model import *
from keras.optimizers import sgd
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets.cifar10 import load_data
import keras
import json
import numpy as np


def lr_reduce(epoch):
    global t
    epoch = t
    if epoch < 100:
        return 1e-2
    elif 100 <= epoch < 175:
        return 1e-3
    else:
        return 1e-4


if __name__ == '__main__':
    try_no = ['resCBoost2']
    models = [ResC]
    for i in range(len(try_no)):
        count = 0
        model = models[i]()
        print(model.summary())
        model.compile(optimizer=sgd(lr=1e-2, momentum=0.9, nesterov=1),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        reduce_lr = LearningRateScheduler(lr_reduce, verbose=1)
        checkpoint = ModelCheckpoint('weights/try-{}.h5'.format(try_no[i]),
                                     monitor='val_acc',
                                     mode='max',
                                     save_best_only=1,
                                     save_weights_only=1)
        tensorboard = TensorBoard(log_dir=os.path.expanduser('log/') + 'yolo' + '_',
                                  histogram_freq=0,
                                  # write_batch_performance=True,
                                  write_graph=True,
                                  write_images=False)
        (x_train, y_t), (x_test, y_te) = load_data()
        y_train = keras.utils.to_categorical(y_t, 10)
        y_test = keras.utils.to_categorical(y_te, 10)
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

        train_generator = ImageDataGenerator(featurewise_center=False,
                                             samplewise_center=False,
                                             featurewise_std_normalization=False,
                                             samplewise_std_normalization=False,
                                             zca_whitening=False,
                                             zca_epsilon=1e-6,
                                             rotation_range=0,
                                             width_shift_range=4,
                                             height_shift_range=4,
                                             shear_range=0,
                                             zoom_range=0,
                                             channel_shift_range=0,
                                             horizontal_flip=True,
                                             vertical_flip=False)
        # train_generator.fit(x_train)
        # test_generator = ImageDataGenerator()
        # train_datagen = train_generator.flow(x_train, y_train, batch_size=128)
        # test_datagen = test_generator.flow(x_test, y_test, batch_size=100)

        # f = model.fit_generator(train_datagen,
        #                         epochs=200,
        #                         validation_data=test_generator, callbacks=[checkpoint, reduce_lr])
        w = np.ones(50000) * 1.
        fs = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': [], 'predict': []}
        terrors = np.zeros(50000)
        for t in range(200):
            f = model.fit(x_train, y_train,
                          epochs=1,
                          batch_size=128,
                          validation_data=[x_test, y_test],
                          callbacks=[reduce_lr],
                          sample_weight=w,
                          # shuffle=False,
                          )
            te = model.predict(x_train)
            for trror in range(50000):
                terrors[trror] = np.argmax(te[trror]) != y_t[trror]
            for key in f.history:
                fs[key].append(f.history[key])
            fs['predict'].append(np.sum(terrors))
            error = 1 - f.history['acc'][-1]
            stage = np.log((1 - error) / error)
            w *= np.exp(stage * terrors)
            w /= np.sum(w) / 50000
            print(np.sum(w))
            with open('log/try_{}.json'.format(try_no[i]), 'w') as wr:
                json.dump(fs.__str__(), wr)
