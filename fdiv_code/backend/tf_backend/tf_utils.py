import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import optimizers

"""
Custom Initializers 
"""

def eye_initializer(shape, dtype=None):
    """
    Initializes a tensor with the identity matrix
    """
    return tf.eye(shape[0], dtype=dtype)

def add_weight_decay(model, weight_decay):
    """
    Adds weight_decay to an already defined model
    """
    if (weight_decay is None) or (weight_decay == 0.0):
        return

    # recursion inside the model
    def add_decay_loss(m, factor):
        if isinstance(m, tf.keras.Model):
            for layer in m.layers:
                add_decay_loss(layer, factor)
        else:
            for param in m.trainable_weights:
                with tf.keras.backend.name_scope('weight_regularizer'):
                    regularizer = lambda: tf.keras.regularizers.l2(factor
                                                                  )(param)
                    m.add_loss(regularizer)

    # weight decay and l2 regularization differs by a factor of 2
    add_decay_loss(model, weight_decay / 2.0)
    return

"""
Optimizers
"""
optimizer_dict = {
    "SGD" : optimizers.SGD,
    "Adam" : optimizers.Adam,
    "Nadam" : optimizers.Nadam,
}

def convert_str_to_optimizer(optimizer_name, **kwargs):
    """
    Loads an optimizer with the given name and specifications
    """
    return optimizer_dict[optimizer_name](**kwargs)

def ilija_schedule(epochs, initial_lr):
    def step_decay(epoch):
        lr = initial_lr * (1 + np.cos(epoch * np.pi / epochs)) / 2
        return lr

    return step_decay

class PeriodicLogger(tf.keras.callbacks.Callback):  
    def __init__(self, every_epoch=100):
        super(PeriodicLogger, self).__init__()
        self.every_epoch = every_epoch
    
    def on_train_begin(self, logs={}):    
        # Initialization code    
        self.epochs = 0    

    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % self.every_epoch == 0:     
            print("-- Epoch {} --".format(self.epochs))
            # Do stuff like printing metrics 
            for log in logs:
                if log == "loss":
                    print("{}: {}".format(log, logs[log]))
                
"""
Tensorflow Custom Functions (used for projected SGD)
"""

def projectOnToEpsilonSimplex_tf(e, epsilon, C):
    """
    Projects a 1D-tensor onto the space of vectors x
    such that x >= epsilon and 1^T x <= C
    """
    d_float = K.cast(tf.shape(e)[0], epsilon.dtype)
    e_eps = e - epsilon
    C_eps = C - epsilon * d_float
    return projectOnToSimplex_tf(e_eps, C_eps) + epsilon

def projectOnToSimplex_tf(e, C):
    """
    Projects a 1D-tensor e onto the space of vectors x
    such that x >= 0 and 1ˆT x <= C
    This Tensorflow function assumes that C >= 0
    Its behavior when C<0 is NOT guaranteed
    """
    d = tf.shape(e)[0]

    e_sorted = tf.sort(e, direction='DESCENDING')
    e_cumsum = tf.math.cumsum(e_sorted)
    
    def find_tau_non_zeros(e_sorted):
        d = tf.shape(e_sorted)[0]
        e_cumsum = tf.math.cumsum(e_sorted)
        
        taus = (e_cumsum - C) / K.cast(tf.range(1, d + 1), e_sorted.dtype)
        ind = K.flatten(tf.where(e_sorted - taus < 0.0))
        ind = tf.cond(tf.equal(tf.shape(ind)[0],0),
                      lambda: d-1,
                      lambda: K.cast(ind[0]-1, d.dtype))
    
        tau = (e_cumsum[ind] - C) / (K.cast(ind, e_sorted.dtype) + 1.0)
        return tau
    
    def find_tau(e_sorted):
        return tf.cond(
            tf.less(K.sum(e_sorted[tf.math.greater(e_sorted,0.0)]),C),
            lambda: 0.0,
            lambda: find_tau_non_zeros(e_sorted)
        )
    
    projected_e = tf.cond(
        tf.math.logical_and(
            tf.less(e_cumsum[d - 1],C),
            tf.less(0.0 , e_sorted[d - 1])
        ),
        lambda: e,
        lambda: tf.maximum(e - find_tau(e_sorted), 0.0)
    )

    return projected_e