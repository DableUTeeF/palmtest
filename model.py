from __future__ import print_function
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import Dense, add, Input
from keras import backend as K, layers


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResBlock(input_tensor, filter, strides=(1, 1)):
    x = Conv2D(filters=filter, kernel_size=(3, 3), padding='same', use_bias=False, strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filter, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if strides != (1, 1):
        input_tensor = Conv2D(filters=filter, kernel_size=1, use_bias=False, strides=strides)(input_tensor)
        input_tensor = BatchNormalization()(input_tensor)
    x = add([input_tensor, x])
    x = Activation('relu')(x)
    return x


def ResBlockV2(input_tensor, filter, strides=(1, 1)):
    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    if strides != (1, 1):
        input_tensor = Conv2D(filters=filter, kernel_size=1, use_bias=False, strides=strides)(x)
        input_tensor = BatchNormalization()(input_tensor)
    x = Conv2D(filters=filter, kernel_size=(3, 3), padding='same', use_bias=False, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filter, kernel_size=(3, 3), padding='same', use_bias=False)(x)
    x = add([input_tensor, x])
    return x


def ResA():
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(32, 5, padding='valid', name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
    x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='c')
    x = conv_block(x, 3, [128, 128, 512], stage=4, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(10, activation='softmax', name='fc')(x)
    return Model(img_input, x, name='resnet')


def ResB():
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(16, 3, padding='same', name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = conv_block(x, 3, [32, 32, 128], stage=3, block='a')
    x = identity_block(x, 3, [32, 32, 128], stage=3, block='b')
    x = conv_block(x, 3, [64, 64, 256], stage=4, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=4, block='b')
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(10, activation='softmax', name='fc')(x)
    return Model(img_input, x, name='resnet')


def ResC():
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(64, 3, padding='same', name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ResBlock(x, 64)
    x = ResBlock(x, 64)
    x = ResBlock(x, 128, (2, 2))
    x = ResBlock(x, 128)
    x = ResBlock(x, 256, (2, 2))
    x = ResBlock(x, 256)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax', name='fc')(x)
    return Model(img_input, x, name='resnet')


def ResF():
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(64, 3, padding='same', name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ResBlockV2(x, 64)
    x = ResBlockV2(x, 64)
    x = ResBlockV2(x, 128, (2, 2))
    x = ResBlockV2(x, 128)
    x = ResBlockV2(x, 256, (2, 2))
    x = ResBlockV2(x, 256)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax', name='fc')(x)
    return Model(img_input, x, name='resnet')


def ResG():
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(16, 3, padding='same', name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ResBlockV2(x, 16)
    x = ResBlockV2(x, 32, (2, 2))
    x = ResBlockV2(x, 64, (2, 2))
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax', name='fc')(x)
    return Model(img_input, x, name='resnet')


def ResD():
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(16, 3, padding='same', name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [16, 16, 64], stage=2, block='b')
    x = conv_block(x, 3, [64, 64, 128], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 128], stage=3, block='b')
    x = conv_block(x, 3, [128, 128, 256], stage=4, block='a')
    x = identity_block(x, 3, [128, 128, 256], stage=4, block='b')
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(10, activation='softmax', name='fc')(x)
    return Model(img_input, x, name='resnet')


def ResE():
    img_input = Input(shape=(32, 32, 3))
    x = Conv2D(64, 3, padding='same', name='conv1')(img_input)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = conv_block(x, 3, [64, 64, 64], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 64], stage=2, block='b')
    x = conv_block(x, 3, [128, 128, 128], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 128], stage=3, block='b')
    x = conv_block(x, 3, [256, 256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 256], stage=4, block='b')
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(10, activation='softmax', name='fc')(x)
    return Model(img_input, x, name='resnet')
