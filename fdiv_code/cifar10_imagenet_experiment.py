import numpy as np
import pandas as pd

import sys
import os
import time
sys.path.insert(1, "backend")

import tensorflow as tf
import tensorflow.keras.backend as K

from tf_backend.tf_models import OneLayerNet, BigConvNet
from tf_backend.tf_losses import pinball_loss_with_scores_keras, dependent_label_quantile_loss_keras, pinball_loss_keras
from tf_backend.tf_utils import *

import np_backend.conformal_utils as cf_utils
from np_backend.dro_conformal import *
from np_backend.numpy_utils import compute_sc_romano_direct_confidence_scores, numpy_softmax
import cvxpy as cp

def run_cqc_experiment(
    data_features,
    data_labels,
    data_scores,
    alpha=0.05,
    data_for_standard_quantile_function="val",
    hidden_layer=True,
    hidden_size=16,
    batch_size_quantile=512,
    epochs_quantile=1000,
    verbose=False,
    add_noise=False,
    variance_noise=0.01,
    **kwargs
):
    """
    Runs the CQC (Conformal Quantile Classification) method
    Fits a quantile function on val data
    Computes an empirical quantile of the scores with/without quantile function on calib data
    Returns coverage and confidence sets on test data for Marginal and CQC methods

    Args:
        data_features: dict containing val / calib / test features
        data_labels : dict containing val / calib / test labels
        data_scores : dict containing val / calib / test scores
        alpha: level of confidence
        data_for_standard_quantile_function: data to use to fit quantile function (default: val)
        hidden_layer / hidden_size: model to use for quantile function (default: 1 layer-NN with 16 hidden neurons)
        epochs_quantile: nb of gd steps for quantile function
        add_noise/variance_noise: whether or not randomize scores in CQC experiment

    Returns:
        confidence_sets: confidence sets for Marginal, CQC and GIQ methods on test data
        coverages: coverage for Marginal, CQC and GIQ methods on test data
        scores_per_method: conformal scores for Marginal, CQC and GIQ methods on calibration and test data
    """
    if "test" in data_features:
        test_set_name = "test"
    elif "testV2" in data_features:
        test_set_name = "testV2"
    else:
        raise Exception("No test set")

    ### Settings
    n_features = data_features["val"].shape[1]
    optimizer_quantile_args = {
        "lr": 1e-4,
        "decay": 1e-5,
        "momentum": 0.9,
        "nesterov": True
    }
    optimizer_quantile = optimizer_dict["SGD"](**optimizer_quantile_args)
    lr_scheduler_quantile = tf.keras.callbacks.LearningRateScheduler(
        ilija_schedule(epochs_quantile, 1e-1)
    )

    ### Quantile Model Method
    noise = {}
    for dataset in ["val", "calib", test_set_name]:
        if add_noise:
            noise[dataset] = variance_noise * np.random.randn(
                data_scores[dataset].shape[0])
        else:
            noise[dataset] = np.zeros(data_scores[dataset].shape[0])

    quantile_cqc_model = cf_utils.quantile_function_fitting(
        data_features[data_for_standard_quantile_function],
        (
            data_scores[data_for_standard_quantile_function] *
            data_labels[data_for_standard_quantile_function]
        ).sum(axis=1, keepdims=True) + noise[
            data_for_standard_quantile_function][:,np.newaxis],
        alpha=alpha,
        optimizer=optimizer_quantile,
        callbacks=[lr_scheduler_quantile],
        epochs_quantile=epochs_quantile,
        batch_size_quantile=batch_size_quantile,
        verbose=verbose,
        hidden_layer=hidden_layer,
        hidden_size=hidden_size
    )

    ### Conformalization
    cqc_quantile_score = {}
    cqc_scores_quantiled = {}

    for dataset in ["val", "calib", test_set_name]:
        cqc_quantile_score[dataset] = quantile_cqc_model.predict(
            x=[
                data_features[dataset],
                (data_scores[dataset] * data_labels[dataset]).sum(axis=1)
            ]
        ).flatten()
        cqc_scores_quantiled[dataset] = (
            data_scores[dataset] * data_labels[dataset]
        ).sum(axis=1) + noise[dataset] - cqc_quantile_score[dataset]

    print("Computing Marginal CS")
    coverage_marginal, conformalized_marginal_scores, Q_marginal = cf_utils.conformalize(
        (data_scores["calib"] * data_labels["calib"]).sum(axis=1),
        (data_scores[test_set_name] * data_labels[test_set_name]).sum(axis=1), alpha
    )
    marginal_confidence_sets = data_scores[test_set_name] - Q_marginal >= 0

    print("Computing CQC CS")
    coverage_cqc, conformalized_cqc_scores, Q_cqc = cf_utils.conformalize(
        cqc_scores_quantiled["calib"], cqc_scores_quantiled[test_set_name], alpha
    )
    cqc_confidence_sets = data_scores[test_set_name] + noise[
        test_set_name][:,np.newaxis] - cqc_quantile_score[
            test_set_name][:, np.newaxis] - Q_cqc >= 0

    print("Computing GIQ CS")
    scores_giq_calib = compute_sc_romano_direct_confidence_scores(
        numpy_softmax(data_scores["calib"]),
        np.random.rand(data_scores["calib"].shape[0])
    )
    scores_giq_test = compute_sc_romano_direct_confidence_scores(
        numpy_softmax(data_scores[test_set_name]),
        np.random.rand(data_scores[test_set_name].shape[0])
    )

    coverage_giq, conformalized_giq_scores, Q_giq = cf_utils.conformalize(
        -(scores_giq_calib * data_labels["calib"]).sum(axis=1),
        -(scores_giq_test * data_labels[test_set_name]).sum(axis=1), alpha
    )
    giq_confidence_sets =  scores_giq_test <= -Q_giq

    confidence_sets = {
        "Marginal": marginal_confidence_sets,
        "CQC": cqc_confidence_sets,
        "GIQ": giq_confidence_sets
    }

    coverages = {
        "Marginal": coverage_marginal,
        "CQC": coverage_cqc,
        "GIQ": coverage_giq
    }

    scores_per_method = {
        "Marginal": {
            "calib" : -data_scores["calib"],
            test_set_name: -data_scores[test_set_name]
        },
        "CQC": {
            "calib" : -data_scores["calib"] + cqc_quantile_score[
                "calib"][:, np.newaxis] - noise["calib"][:,np.newaxis],
            test_set_name: -data_scores[test_set_name] + cqc_quantile_score[
                test_set_name][:, np.newaxis] - noise[test_set_name][:,np.newaxis]
        },
        "GIQ": {
            "calib": scores_giq_calib,
            test_set_name: scores_giq_test
        }
    }

    return confidence_sets, coverages, scores_per_method

def initialize_df_multitrial_summary(conformal_list, robust_list, n_trials, n_classes):
    """
    Returns an empty dataframe of the form
    "Conformalization" | "Robustness" | "AvgSetSize" | "Coverage" | "Rho"
        Marginal           Naive             NA            NA         NA
        CQC                Naive             NA            NA         NA

        Marginal           Naive             NA            NA         NA
        CQC                Naive             NA            NA         NA
    """
    df_multitrial_summary = pd.DataFrame(
            index=np.arange(len(conformal_list) * len(robust_list) * n_trials),
            columns=["Conformalization", "Robustness", "AvgSetSize", "Coverage", "Rho"]
        + list(np.char.add('Size', np.arange(n_classes + 1).astype(str)))
        + list(np.char.add('SizeCoverage', np.arange(n_classes + 1).astype(str)))
    )
    df_multitrial_summary["Robustness"] = np.repeat(
        np.tile(robust_list, n_trials), len(conformal_list))
    df_multitrial_summary["Conformalization"] = np.tile(conformal_list, n_trials * len(robust_list))
    df_multitrial_summary.iloc[:, -2 * (n_classes + 1):] = 0
    return df_multitrial_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    path_to_scratch = os.getcwd()

    parser.add_argument('--nTrials', default=20, type=int)
    parser.add_argument('--alpha_coverage', default=0.05, type=float)
    parser.add_argument('--dataset', default="cifar-10", type=str)
    parser.add_argument('--n_slabs_directions', default=1000, type=int)
    parser.add_argument('--delta_slab', default=0.1, type=float)
    parser.add_argument('--alpha_slab', default=0.1, type=float)
    parser.add_argument('--rho_validation', default="slab_quantile", type=str)
    parser.add_argument('--experiment_name', default="debug", type=str)

#     Parameters for CQC fitting method
    parser.add_argument('--cqc_epochs_quantile', default=10000, type=int)
    parser.add_argument('--cqc_batchsize', default=512, type=int)
    parser.add_argument('--cqc_add_noise', help='Noise in CQC', action='store_true')
    parser.add_argument('--cqc_variance_noise', default=0.1, type=float)

    args = parser.parse_args()
    print("Arguments: ", args)

    alpha = args.alpha_coverage
    chisq = lambda z: 0.5 * cp.sum_squares(z - 1)  # This is the chi-squared ball.

    path_to_experiment_folder = os.path.join(
        path_to_scratch, "experiments/robust_cv", args.dataset.lower())

    if args.dataset == "cifar-10":
        data_folder = os.path.join("datasets", args.dataset.lower())
    elif args.dataset == "imagenet":
        data_folder = os.path.join("datasets", "ImageNet")
    else:
        raise NotImplementedError("Unknown Dataset")

    path_to_data = os.path.join(path_to_scratch,data_folder)
    if not os.path.exists(path_to_experiment_folder):
        os.mkdir(path_to_experiment_folder)
    path_to_experiment_files = os.path.join(path_to_experiment_folder,args.experiment_name)

    print("***** STARTING MULTICLASS EXPERIMENT WITH DATASET {} ****".format(args.dataset))

    conformal_algos = ["Marginal", "CQC", "GIQ"]

    if args.dataset == "cifar-10":
        dataset_split_list = ["train", "val", "testV2"]
    elif args.dataset == "imagenet":
        dataset_split_list = ["val", "test"]
    else:
        raise NotImplementedError("Unknown Dataset")

    print("*** Loading Data ***")
    data = {}
    for dataset in dataset_split_list:
        data[dataset] = np.load(path_to_data + "/np_data/" + dataset + ".npy",
                                allow_pickle=True).item()

    df_multitrial_summary = initialize_df_multitrial_summary(
        conformal_algos,
        ["NaiveConformal", "RobustConformal"],
        args.nTrials,
        data["val"]["labels"].shape[1]
    )

    print(df_multitrial_summary)

    index = 0
    for M in range(args.nTrials):
        print("*** TRIAL {} ****".format(M))
        K.clear_session()

        if args.dataset == "cifar-10":
            split_percentages = np.array([70, 30])
        elif args.dataset == "imagenet":
            split_percentages = np.array([80,20])

        # Randomly split between validation / calibration
        val_permutation_split, val_splitted_arrays = cf_utils.split_conformal(
            [data["val"]["features"], data["val"]["labels"],  data["val"]["scores"]],
            split_percentages=split_percentages
        )
        val_splitted_features, val_splitted_labels, val_splitted_scores = val_splitted_arrays


        data_features = dict(zip(["val", "calib"], val_splitted_features))
        data_labels = dict(zip(["val", "calib"], val_splitted_labels))
        data_scores = dict(zip(["val", "calib"], val_splitted_scores))

        data_features[dataset_split_list[-1]] = data[dataset_split_list[-1]]["features"]
        data_labels[dataset_split_list[-1]]   = data[dataset_split_list[-1]]["labels"]
        data_scores[dataset_split_list[-1]]   = data[dataset_split_list[-1]]["scores"]

#         print("Training Set Size: {}".format(data_features["train"].shape[0]))
        print("Validation Set Size: {}".format(data_features["val"].shape))
        print("Calibration Set Size: {}".format(data_features["calib"].shape))
        print("Testing Set Size: {}".format(data_features[dataset_split_list[-1]].shape))

        # Run Conformal Methods
        naive_confidence_sets, naive_coverages, scores_per_method = run_cqc_experiment(
            data_features,
            data_labels,
            data_scores,
            alpha=alpha,
            data_for_standard_quantile_function="val",
            epochs_quantile=args.cqc_epochs_quantile,
            batch_size_quantile=args.cqc_batchsize,
            add_noise=args.cqc_add_noise,
            variance_noise=args.cqc_variance_noise,
            verbose=True,
        )

        for conformal_method in conformal_algos:
            confidence_set_sizes = naive_confidence_sets[conformal_method].sum(axis=1)
            coverage_per_instance = (
                naive_confidence_sets[conformal_method] * data_labels[dataset_split_list[-1]]
            ).sum(axis=1)>0
            unique_sizes, unique_size_counts = np.unique(
                confidence_set_sizes, return_counts=True)
            unique_sizes_coverage, unique_size_counts_coverage = np.unique(
                confidence_set_sizes[coverage_per_instance],
                return_counts=True)

            dict_sizes = dict(
                zip(unique_sizes, unique_size_counts))
            dict_sizes_coverage = dict(
                zip(unique_sizes_coverage, unique_size_counts_coverage)
            )

            df_multitrial_summary["AvgSetSize"][index] = np.array(
                naive_confidence_sets[conformal_method], dtype=bool
            ).sum(axis=1).mean()
            df_multitrial_summary["Coverage"][index] = naive_coverages[conformal_method]

            for size in dict_sizes:
                df_multitrial_summary['Size' + str(size)][index] = dict_sizes[size]
            for size in dict_sizes_coverage:
                df_multitrial_summary['SizeCoverage' +
                                      str(size)][index] = dict_sizes_coverage[size]
            index+=1

        slab_quantiles = {
            "Marginal": np.zeros(args.n_slabs_directions),
            "CQC": np.zeros(args.n_slabs_directions),
            "GIQ":  np.zeros(args.n_slabs_directions)
        }
        q_slab = {}

        conformal_scores = {}
        for conformal_method in conformal_algos:
            conformal_scores[conformal_method] = (
                scores_per_method[conformal_method]["calib"] * data_labels["calib"]
            ).sum(axis=1)


        # Run robust validation with slabs
        if args.rho_validation == "slab_quantile":
            for slab_idx in range(args.n_slabs_directions):
                if slab_idx % 10 == 0:
                    print("Slab Id={}".format(slab_idx))
                # Might want to try out different sampling mechanisms for direction
                direction = np.random.randn(data_features["calib"].shape[1])
                direction = direction / np.linalg.norm(direction)

                for conformal_method in conformal_algos:
                    slab_quantiles[conformal_method][slab_idx] = find_worst_case_slab_quantile(
                        direction, data_features["calib"],
                        conformal_scores[conformal_method], alpha, args.delta_slab)

            # Computing the new robust threshold
            for conformal_method in conformal_algos:
                q_slab[conformal_method] = np.quantile(
                    slab_quantiles[conformal_method], 1 - args.alpha_slab, interpolation="higher")

        elif args.rho_validation in ["learnable_direction_OLS","learnable_direction_SVM"]:
            for conformal_method in conformal_algos:
                q_slab[conformal_method] = learnable_direction_quantile(
                        data_features["calib"],
                        conformal_scores[conformal_method],
                        np.arange(data_features["calib"].shape[0]),
                        model=args.rho_validation.split("_")[-1],
                        alpha=alpha,
                        delta=args.delta_slab
                    )
        else:
            raise NotImplementedError("Rho Validation Method Not Implemented")

        robust_confidence_sets = {}

        for conformal_method in conformal_algos:
            robust_confidence_sets[conformal_method] = scores_per_method[conformal_method][
                dataset_split_list[-1]] <= q_slab[conformal_method]

            confidence_set_sizes = robust_confidence_sets[conformal_method].sum(axis=1)
            coverage_per_instance = (
                robust_confidence_sets[conformal_method] * data_labels[dataset_split_list[-1]]
            ).sum(axis=1)>0

            unique_sizes, unique_size_counts = np.unique(
                confidence_set_sizes, return_counts=True)
            unique_sizes_coverage, unique_size_counts_coverage = np.unique(
                confidence_set_sizes[coverage_per_instance],
                return_counts=True)

            dict_sizes = dict(
                zip(unique_sizes, unique_size_counts))
            dict_sizes_coverage = dict(
                zip(unique_sizes_coverage, unique_size_counts_coverage)
            )


            df_multitrial_summary["AvgSetSize"][index] = np.array(
                robust_confidence_sets[conformal_method], dtype=bool
            ).sum(axis=1).mean()
            df_multitrial_summary["Coverage"][index] = (
                robust_confidence_sets[conformal_method] * data_labels[dataset_split_list[-1]]
            ).sum(axis=1).mean()
            # What rho does it correspond to?
            df_multitrial_summary["Rho"][index] = find_rho_for_quantile(
                conformal_scores[conformal_method], q_slab[conformal_method],
                chisq, alpha=alpha, delta=args.delta_slab
            )
            for size in dict_sizes:
                df_multitrial_summary['Size' + str(size)][index] = dict_sizes[size]
            for size in dict_sizes_coverage:
                df_multitrial_summary['SizeCoverage' +
                                      str(size)][index] = dict_sizes_coverage[size]
            index+= 1

    print("*** Saving Results ***")

    df_multitrial_summary.to_csv(
        path_to_experiment_files + "-summary.csv")
