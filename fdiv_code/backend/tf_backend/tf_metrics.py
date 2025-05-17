import tensorflow as tf
import tensorflow.keras.backend as K


"""
Metrics for multilabel classifications
"""
def recall_keras(y_true, y_pred):
    """
    Returns TP / (TP + FN)
    where TP = #{y_true = 1 AND y_pred > 0.5}
          FN = #{y_true = 1 AND y_pred < 0.5}
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_keras(y_true, y_pred):
    """
    Returns TP / (TP + FP)
    where TP = #{y_true = 1 AND y_pred > 0.5}
          FN = #{y_true = 0 AND y_pred > 0.5}
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_keras(y_true, y_pred):
    """
    Returns the F1-Score of the classifier
    where F1 = 2 * (precision * recall) / (precision + recall)
    """
    precision = precision_keras(y_true, y_pred)
    recall = recall_keras(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))