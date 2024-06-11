import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Input, Dropout, GlobalAveragePooling2D

IMG_SIZE = (32, 32)

preprocess_input = tf.keras.applications.vgg19.preprocess_input

# building VGG19 Model from scratch
def cifar10_custom_VGG19(image_shape=IMG_SIZE, weights=False):
    ''' Define a tf.keras model for 10 categories classification using VGG19 model structure
    Arguments:
        image_shape -- Image width and height
        weights -- weights for the model
    Returns:
        tf.keras.model
    '''
    # defining the input shape
    input_shape = image_shape + (3,)
    
    # defining the input layer
    inputs = Input(shape=input_shape, name='input_layer')

    # Preprocess inputs
    x = preprocess_input(inputs)
    
    # creating VGG19 blocks
    # block1: 2 convolutions, 1 maxpooling
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block2: 2 convolutions, 1 maxpooling
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # block3: 4 convolutions, 1 maxpooling
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # block4: 4 convolutions, 1 maxpooling
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # block5: 4 convolutions, 1 maxpooling
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Batchnormalization
    x = BatchNormalization(name='batch_norm')(x)
    
    # flattening the output of the convolutional layers
    x = GlobalAveragePooling2D(name='global_average')(x)
    # including dropout to avoid overfitting
    x = Dropout(0.3, name='global_dropout')(x)

    # fully connected layers and Dropout layers
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.3, name='fc1_dropout')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dropout(0.2, name='fc2_dropout')(x)
    x = Dense(1024, activation='relu', name='fc3')(x)
    x = Dropout(0.2, name='fc3_dropout')(x)
    x = Dense(1024, activation='relu', name='fc4')(x)
    x = Dropout(0.2, name='fc4_dropout')(x)
    
    # output layer
    outputs = Dense(10, activation='softmax', name='predictions')(x)
    
    # creating the model
    model = Model(inputs=inputs, outputs=outputs)

    # assigning weights if given
    if weights != False:
      model.load_weights(weights)
    
    return model
