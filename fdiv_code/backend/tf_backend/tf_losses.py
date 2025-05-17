import tensorflow as tf
import tensorflow.keras.backend as K

from tf_backend.tf_layers import Quad_Over_Lin

"""
Losses used in quantile regression and classification
"""
def pinball_loss_keras(x, alpha=0.05):
    return (1 - alpha) * K.maximum(-x, 0.0) + alpha * K.maximum(x, 0.0)


def quantile_loss_keras(alpha=0.05):
    def loss(y_true, y_pred):
        return K.mean(
            pinball_loss_keras(K.sum(y_pred * y_true, axis=1), alpha=alpha)
        )

    return loss


def pinball_loss_with_scores_keras(alpha=0.05):
    def loss(y_true, y_pred):
        return K.mean(
            pinball_loss_keras(y_true-y_pred, alpha=alpha)
        )
    return loss

def combined_loss_with_scores_keras(alpha=0.05):
    def combined_loss(y_true, y_pred):
        scores_inner, scores_outer = y_true
        quantile_inner, quantile_outer = y_pred
        return K.mean(
            pinball_loss_keras(
                K.minimum(scores_outer-quantile_outer, quantile_inner-scores_inner),
            alpha=alpha)
        )
    return combined_loss


def zero_loss_keras(y_true, y_pred):
    return K.zeros_like(y_pred)


def dependent_label_quantile_loss_keras(
    label_tensor,
    score_tensor,
    quantile_tensor,
    n_classes=20,
    alpha = 0.05,
    epsilon = 0.01,
    max_trace = 10**4,
    trainable_sigma=True
):
        
    error = label_tensor - score_tensor    
    mahalanobis_distance = Quad_Over_Lin(n_classes, epsilon, max_trace, trainable_sigma)(error)
    
    individual_loss = K.maximum(
        mahalanobis_distance - quantile_tensor, 0.0
    ) + alpha * quantile_tensor
    
    return K.mean(individual_loss)
