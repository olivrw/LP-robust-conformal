import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.constraints import Constraint

from tf_backend.tf_utils import projectOnToEpsilonSimplex_tf

"""
Constraints used in Keras
"""
class PSD_With_Trace(Constraint):
    """
    Implements the following constaint on a square matrix Sigma
    - Sigma is symmetric
    - Sigma >= epsilon * I (in the PSD sense)
    - trace(Sigma) <= max_trace
    """
    def __init__(self, epsilon=0.01, max_trace=10**4):
        self.max_trace = tf.constant(max_trace, dtype=tf.float32)
        self.epsilon   = tf.constant(epsilon, dtype=tf.float32)
        

    def __call__(self, Sigma):
        # Ensure Sigma is symmetric
        Sigma_sym = 0.5 * (Sigma + tf.transpose(Sigma))
        
        # Compute the eigenvalue decomposition
        e,v = tf.linalg.eigh(Sigma_sym)
        
        # Project onto the space of PSD matrix with eigenvalue >= epsilon
        # and trace <= max_trace
        e_transformed = projectOnToEpsilonSimplex_tf(e, self.epsilon, self.max_trace)
        return (v * e_transformed) @ tf.transpose(v)

    def get_config(self):
        return {
            'epsilon': self.epsilon,
            'max_trace': self.maximum
        }
