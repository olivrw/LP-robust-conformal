import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.constraints import MinMaxNorm

from tf_backend.tf_utils import add_weight_decay

# Pretrained Models
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2, ResNet50V2
from tensorflow.keras.applications import InceptionResNetV2

def OneLayerNet(
    input_tensor=None,
    input_shape=2048,
    prior_processing_tensor=None,
    output_shape=500,
    n_outputs=1,
    flatten_outputs=True,
    label_tensors_as_input=False,
    hidden_layer=False,
    hidden_size=1024,
    final_activation=tf.nn.sigmoid,
    **kwargs
):
    """
    Build a multi-label model with zero or one hidden layer.
    """
    if input_tensor is None:
        input_tensor = Input(input_shape)
    if prior_processing_tensor is not None:
        x_input = prior_processing_tensor
    else:
        x_input = input_tensor
    
    outputs = []
    for _ in range(n_outputs):
        if hidden_layer:
            x = Dense(hidden_size, activation=tf.nn.relu)(x_input)
        else:
            x = x_input
        x = Dense(output_shape, activation=final_activation)(x)
        if output_shape == 1 & flatten_outputs:
            x = tf.reshape(x, [-1])
        outputs.append(x)
    
    if label_tensors_as_input:
        input_tensor = [input_tensor]
        for _ in range(n_outputs):
            new_input = Input(output_shape)
            input_tensor.append(new_input)
    return Model(inputs=input_tensor, outputs=outputs)

def BigConvNet(name="ResNet50",
                input_tensor=None,
                input_shape=(224, 224, 3),
                output_shape=20,
                classification_type="multilabel",
                retrain=True,
                dropout=True,
                dropout_rate=0.7,
                l2_reg=0.0,
                retrain_from_layer=0,
               include_top=False,
                **kwargs):
    """
    Returns a ResNet50 network pre-trained on ImageNet
    Possibly adds a dropout layer before the output layer
    Retrains only the last "blocks_to_retrain" ResNet blocks
    """
    kwargs_base_model = {
        "input_tensor" : input_tensor,
        "weights" : "imagenet",
        "include_top" : include_top,
        "input_shape" : input_shape,
        "pooling" : 'avg'
    }
    
    if name == "ResNet50":
        base_model = ResNet50(
            **kwargs_base_model
        )
    elif name == "ResNet50V2":
        base_model = ResNet50V2(
            **kwargs_base_model
        )
    elif name == "MobileNetV2":
        base_model = MobileNetV2(
            **kwargs_base_model
        )
    elif name == "ResNet101V2":
        base_model = ResNet101V2(
            **kwargs_base_model
        )
        
    elif name == "InceptionResNetV2":
        base_model = InceptionResNetV2(
            **kwargs_base_model
        )
        
    elif name == "ResNetCIFARV2":
        depth = kwargs["n_blocks"] * 9 + 2
        return resnet_v2(input_shape=input_shape, depth=depth, num_classes=output_shape)
    elif name == "ResNetCIFAR":
        depth = kwargs["n_blocks"] * 6 + 2
        return resnet_v1(input_shape=input_shape, depth=depth, num_classes=output_shape)
    else:
        raise NotImplemented()
        
    print("Model pre-loaded")
    
    x = base_model.output
    
    if not(include_top):
        if dropout:
            x = Dropout(dropout_rate)(x)

        if classification_type == "multiclass":
            x = Dense(output_shape, activation=tf.nn.softmax, name="output_layer")(x)
        elif classification_type == "multilabel":
            x = Dense(output_shape, activation=tf.nn.sigmoid, name="output_layer")(x)
        else:
            x = Dense(output_shape, activation=None, name="output_layer")(x)

        model = Model(inputs=base_model.input, outputs=x)
    
    else:
        model = base_model
        
    for layer in base_model.layers:
        layer.trainable = False
    if retrain:
        for layer in base_model.layers[retrain_from_layer:]:
            layer.trainable = True
            
    add_weight_decay(model, l2_reg)
    return model


def ResNet50Net(**kwargs):
    return BigConvNet(name="ResNet50", **kwargs)

def QuantileModel(
    prediction_model,
    quantile_model,
    loss_function,
    nClasses=100,
    train_prediction_model=False,
    **quantile_params
):
    """
    Model which fits a quantile function as a given function of the output, 
    as well as label correlation matrix sigma
    """
    
    # Verify that the number of classes matches
    assert(nClasses == prediction_model.output.shape[1])
    
    # Verify that both inpus are the same
    assert((prediction_model.inputs[0].name == quantile_model.inputs[0].name) & 
           (prediction_model.inputs[0].graph == quantile_model.inputs[0].graph)
          )

    feature_input = prediction_model.inputs[0]
    score_output = prediction_model.output
    quantile_output = quantile_model.output

    if not (train_prediction_model):
        for layer in prediction_model.layers:
            layer.trainable = False
      
    label_input = Input(nClasses, name="input_label")    

    
    full_model = Model(inputs=[feature_input, label_input],
                      outputs=[score_output, quantile_output])
    
    quantile_loss = loss_function(label_input, score_output, quantile_output, nClasses=nClasses, **quantile_params)
 
    full_model.add_loss(quantile_loss)

    return full_model

    
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    model_ResNet = ResNet50Net()
    print(model_ResNet.summary())

    model_One_Layer = OneLayerNet()
    print(model_One_Layer.summary())