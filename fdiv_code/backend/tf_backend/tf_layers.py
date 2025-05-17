import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

from tf_backend.tf_constraints import PSD_With_Trace
from tf_backend.tf_utils import eye_initializer

"""
Layers and initializers used for Keras
"""
class Quad_Over_Lin(layers.Layer):
    """
    Builds a layer which takes in input a vector error 
    """
    def __init__(self, nClasses=500, epsilon=0.01, max_trace=10**4, trainable_sigma=True):
        super(Quad_Over_Lin, self).__init__()
        self.nClasses = nClasses
        self.epsilon = epsilon
        self.max_trace = max_trace
        self.trainable_sigma= trainable_sigma
        if self.max_trace - self.nClasses * self.epsilon < 0:
            raise ValueError("The maximum trace is not sufficiently high compared to epsilon")

    def build(self, input_shape):
        self.sigma = self.add_weight(
            shape=(self.nClasses, self.nClasses),
            initializer=eye_initializer,
            trainable=self.trainable_sigma,
            constraint=PSD_With_Trace(self.epsilon, self.max_trace)
        )

    def call(self, inputs):
        return K.sum(
            layers.Multiply()(
                [
                    inputs,
                    tf.transpose(tf.linalg.solve(self.sigma, tf.transpose(inputs))),
                ]
            ),
            axis=1
        )

    def get_config(self):
        config = super(Quad_Over_Lin, self).get_config()
        config.update({'nClasses': self.nClasses})
        return config

