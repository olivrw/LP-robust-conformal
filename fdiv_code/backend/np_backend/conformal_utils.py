import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from tensorflow.keras.utils import to_categorical

from tf_backend.tf_models import OneLayerNet
from tf_backend.tf_losses import pinball_loss_with_scores_keras
from tf_backend.tf_utils import PeriodicLogger

            
def conformalize(scores_calibration, scores_testing, alpha):
    n = scores_calibration.shape[0]
    
    Q = - np.quantile(-scores_calibration, (1-alpha) * (1.0 + 1.0 / n),
                      interpolation="higher")

    coverage_test = np.mean(scores_testing - Q > 0)
    conformalized_scores_testing = scores_testing - Q

    return coverage_test, conformalized_scores_testing, Q


def split_conformal(list_of_arrays, split_percentages=np.array([50, 30, 20])):
    N = list_of_arrays[0].shape[0]
    permutation = np.random.permutation(N)
    splitted_arrays = []

    assert (np.sum(split_percentages) == 100)

    print(np.floor(split_percentages.cumsum() / 100.0 * N).astype(int)[:-1])

    for array in list_of_arrays:
        array_split = np.split(
            array[permutation],
            np.floor(split_percentages.cumsum() / 100.0 * N).astype(int)[:-1]
        )
        splitted_arrays.append(array_split)
    return permutation, splitted_arrays


def get_per_label_coverage(confidence_sets, labels, n_classes=None):
    if labels.ndim == 1:
        labels = to_categorical(labels)
    n_classes = labels.shape[1]
    coverage_per_label = np.zeros(n_classes)
    for k in range(n_classes):
        coverage_per_label[k] = confidence_sets[labels[:,k]==1,k].mean()
    return coverage_per_label


def compute_tree_structured_quantiled_score(
    labels,
    scores,
    prediction_graph,
    features=None,
    quantile_model=None,
    quantile_scores_per_example=None
):
    if quantile_scores_per_example is not None:
        return prediction_graph.compute_log_likelihood(
            labels, scores
        ) - quantile_scores_per_example

    return prediction_graph.compute_log_likelihood(
        labels, scores
    ) - quantile_model.predict(features).flatten()

def compute_tree_inner_set(
    prediction_graph,
    scores,
    Q_conformal=0.0,
    features=None,
    quantile_model=None,
    quantile_scores_per_example=None,
):
    #     For each class k, get the most likely state with y_k=0 and check if log p_theta(y|f(x)) < q(x)
    inner_sets = np.zeros_like(scores)
    n_classes = scores.shape[1]
        
    for k in range(n_classes):
        fake_labels, _ = prediction_graph.get_most_likely_configuration(
            scores, k, 0
        )
        inner_sets[:, k] = compute_tree_structured_quantiled_score(
            fake_labels,
            scores,
            prediction_graph,
            features=features,
            quantile_model=quantile_model,
            quantile_scores_per_example=quantile_scores_per_example
        ) - Q_conformal < 0
        
    return inner_sets

def compute_tree_outer_set(
    prediction_graph,
    scores,
    Q_conformal=0.0,
    features=None,
    quantile_model=None,
    quantile_scores_per_example=None,
):
    #     For each class k, get the most likely state with y_k=1 and check if log p_theta(y|f(x)) <= q(x)
    outer_sets = np.zeros_like(scores)
    n_classes = scores.shape[1]

    for k in range(n_classes):
        fake_labels, _ = prediction_graph.get_most_likely_configuration(
            scores, k, 1
        )
        outer_sets[:, k] = 1 - (
            compute_tree_structured_quantiled_score(
                fake_labels,
                scores,
                prediction_graph,
                features=features,
                quantile_model=quantile_model,
                quantile_scores_per_example=quantile_scores_per_example
            ) - Q_conformal < 0
        )

    return outer_sets

def compute_tree_inner_outer_sets(**kwargs):
    outer_sets =  compute_tree_outer_set(**kwargs)
    inner_sets = compute_tree_inner_set(**kwargs)
    return inner_sets * outer_sets, outer_sets

def compute_maximum_weight_spanning_tree(I_Y_X):
    """
    I_Y_X = E[Y_i,Y_j | X]
    """
    I_Y_X = I_Y_X - np.diag(np.diag(I_Y_X))
    I_Y_X = csr_matrix(I_Y_X)
    T_max_spanning = minimum_spanning_tree(-I_Y_X)

    return T_max_spanning


def quantile_function_fitting(features, scores, alpha, optimizer, callbacks, 
                              epochs_quantile, batch_size_quantile=None, verbose=False, **kwargs):
    n_features = features.shape[1]
    
    quantile_model = OneLayerNet(
            input_shape=n_features,
            output_shape=1,
            n_outputs=1,
            final_activation=None,
            flatten_outputs=False,
            **kwargs
        )
    
    ### Quantile Function Optimization

    quantile_model.compile(
        optimizer, loss=pinball_loss_with_scores_keras(alpha=alpha)
    )
    if verbose:
        print(quantile_model.summary())
        
    if verbose:
        print("Evaluation of an alpha={} quantile model before training: {}".format(
            alpha,
            quantile_model.evaluate(
                x=features,
                y=scores,
                batch_size=features.shape[0],
                verbose=False
                )
            )
        )
        
    if batch_size_quantile is None:
        batch_size_quantile = features.shape[0]
    quantile_model.fit(
        x=features, 
        y=scores,
        batch_size=batch_size_quantile,
        epochs=epochs_quantile,
        callbacks=callbacks + [PeriodicLogger(1000)],
    )
    if verbose:
        print("Evaluation of an alpha={} quantile model after training: {}".format(
            alpha,
            quantile_model.evaluate(
                x=features, 
                y=scores,
                batch_size=features.shape[0],
                verbose=False
                )
            )
        )
    return quantile_model

def get_worse_slab_coverage(coverage_sorted_per_projecion, delta=0.2):
    n_test = len(coverage_sorted_per_projecion)
    min_size = int(n_test * delta)

    min_coverage = 1.0
    min_interval = (0, n_test - 1)

    N_covered = np.zeros((n_test, n_test))
    for i in range(n_test):
        for j in np.arange(i+min_size, n_test):
            N_covered[i, j] = np.mean(coverage_sorted_per_projecion[i:j + 1])
            if (N_covered[i, j] < min_coverage):
                min_coverage = N_covered[i, j]
                min_interval = (i, j)

    return min_coverage, (i, j)


def get_conditional_coverage_from_inout_sets(
    conditional_probabilities, complete_label_configurations, inner_sets,
    outer_sets
):
    io_confidence_sets = np.zeros(
        (inner_sets.shape[0], complete_label_configurations.shape[0])
    )

    for config in range(complete_label_configurations.shape[0]):
        y = (complete_label_configurations[config] > 0)
        io_confidence_sets[:, config] = np.all(
            y[np.newaxis, :] >= inner_sets, axis=1
        ) * np.all(y[np.newaxis, :] <= outer_sets, axis=1)

    return (conditional_probabilities * io_confidence_sets).sum(axis=1)