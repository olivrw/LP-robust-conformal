import tensorflow as tf

from tf_backend.tf_models import *

model_dict = {
    "OneLayerNet": OneLayerNet,
    "ResNet50PreTrainedNet":ResNet50PreTrainedNet,
    "ResNet50Net":ResNet50Net,
    "QuantileModel" : QuantileModel}

def load_model(model_name, input_shape, output_shape, **kwargs):
    """
    Loads a net with the given name and options specified with kwargs
    """
    return model_dict[model_name](intput_shape=input_shape, output_shape=output_shape, **kwargs)

def unfreeze_layers(model, from_layer=143):
    for layer in model.layers[from_layer:]:
        layer.trainable = True
    