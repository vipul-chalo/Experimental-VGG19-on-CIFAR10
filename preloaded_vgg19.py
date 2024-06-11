import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg19 import VGG19

IMG_SIZE = (32, 32)

preprocess_input = tf.keras.applications.vgg19.preprocess_input

def cifar10_VGG19(image_shape=IMG_SIZE, if_trainable=False, layers=1):
    ''' Define a tf.keras model for 10 categories classification out of the VGG19 model
    Arguments:
        image_shape -- Image width and height
        if_trainable -- True or False
        layers -- Number of layers of base_model that are trainable (max=22)
    Returns:
        tf.keras.model
    '''
    input_shape = image_shape + (3,)

    # defining VGG19 base model
    base_model = VGG19(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # freezing the base model initially by making it non trainable
    base_model.trainable = if_trainable

    # creating the input layer
    inputs = tf.keras.Input(shape=input_shape) 
    
    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(inputs) 
    
    # setting training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False) 

    # Batchnormalization
    x = BatchNormalization(name='batch_norm')(x)
    
    # adding the new classification layers
    # using global avg pooling to summarize the info in each channel
    x = GlobalAveragePooling2D(name='global_pool')(x) 
    # including dropout to avoid overfitting
    x = Dropout(0.3, name='global_dropout')(x)

    # including fully connected layers with dropout
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.3, name='fc1_dropout')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    x = Dropout(0.2, name='fc2_dropout')(x)
    x = Dense(1024, activation='relu', name='fc3')(x)
    x = Dropout(0.2, name='fc3_dropout')(x)
    x = Dense(1024, activation='relu', name='fc4')(x)
    x = Dropout(0.2, name='fc4_dropout')(x)
    
    # using a prediction layer
    outputs = Dense(10, activation='softmax', name='predictions')(x)

    # defining the model
    model = Model(inputs, outputs)
  
    if if_trainable == True:
      base_model = model.layers[3]
      base_model.trainable = True
      
      # fine-tuning from this layer onwards
      fine_tune_at = len(base_model.layers) - layers
      
      # freezing all the layers before the `fine_tune_at` layer
      for layer in base_model.layers[:fine_tune_at]:
          layer.trainable = False
    
    return model

