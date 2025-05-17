import numpy as np
import pandas as pd

import sys
import os
import time
sys.path.insert(1, "./backend")
sys.path.insert(1, "./experiments")

import tensorflow as tf
import tensorflow.keras.backend as K

from tf_backend.tf_models import OneLayerNet, BigConvNet
from tf_backend.tf_losses import pinball_loss_with_scores_keras, dependent_label_quantile_loss_keras, pinball_loss_keras
from tf_backend.tf_utils import *

import np_backend.conformal_utils as cf_utils
from np_backend.dro_conformal import *

import cvxpy as cp

def run_cqc_experiment(
    data_features,
    data_labels,
    data_scores,
    alpha=0.05,
    data_for_standard_quantile_function="val",
    hidden_layer=True,
    hidden_size=16,
    epochs_quantile=1000,
    verbose=False,
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

    Returns:
        cqc_confidence_sets: confidence sets for CQC method on test data
        marginal_confidence_sets: confidence sets for Marginal (Naive conformal) method on test data
        coverage_cqc: coverage for CQC method on test data
        coverage_marginal: coverage for Marginal method on test data
        cqc_quantile_score: vector of q(x) where x belongs to the test data: the final scores for CQC are s(x,y)-q(x)
    """
    if "test" in data_features:
        test_set_name = "test"
    elif "testV2" in data_features:
        test_set_name = "testV2"
    else:
        raise Exception("No test set")
        
    ### Settings
    n_features = data_features["train"].shape[1]
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
    quantile_cqc_model = cf_utils.quantile_function_fitting(
        data_features[data_for_standard_quantile_function],
        (
            data_scores[data_for_standard_quantile_function] *
            data_labels[data_for_standard_quantile_function]
        ).sum(axis=1, keepdims=True),
        optimizer_quantile, [lr_scheduler_quantile],
        epochs_quantile,
        alpha,
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
        ).sum(axis=1) - cqc_quantile_score[dataset]

    coverage_cqc, conformalized_cqc_scores, Q_cqc = cf_utils.conformalize(
        cqc_scores_quantiled["calib"], cqc_scores_quantiled[test_set_name], alpha
    )

    coverage_marginal, conformalized_marginal_scores, Q_marginal = cf_utils.conformalize(
        (data_scores["calib"] * data_labels["calib"]).sum(axis=1),
        (data_scores[test_set_name] * data_labels[test_set_name]).sum(axis=1), alpha
    )

    cqc_confidence_sets = data_scores[test_set_name] - cqc_quantile_score[
        test_set_name][:, np.newaxis] - Q_cqc > 0

    marginal_confidence_sets = data_scores[test_set_name] - Q_marginal > 0

    return cqc_confidence_sets, marginal_confidence_sets, coverage_cqc, coverage_marginal, cqc_quantile_score

def initialize_df_multitrial_summary(conformal_list, robust_list, n_trials):
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
        )
    df_multitrial_summary["Robustness"] = np.repeat(
        np.tile(robust_list, n_trials), len(conformal_list))
    df_multitrial_summary["Conformalization"] = np.tile(conformal_list, n_trials * len(robust_list))
    return df_multitrial_summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    path_to_scratch = os.getcwd() + '/'

    parser.add_argument('--nTrials', default=20, type=int)
    parser.add_argument('--alpha_coverage', default=0.05, type=float)
    parser.add_argument('--dataset', default="CIFAR-10", type=str)
    parser.add_argument('--epochs_quantile', default=10000, type=int)
    parser.add_argument('--n_slabs_directions', default=1000, type=int)
    parser.add_argument('--delta_slab', default=0.1, type=float)
    parser.add_argument('--alpha_slab', default=0.1, type=float)
    parser.add_argument('--rho_validation', default="slab_quantile", type=str)
    parser.add_argument('--experiment_name', default="debug", type=str)
    
    args = parser.parse_args()
    alpha = args.alpha_coverage
    chisq = lambda z: 0.5 * cp.sum_squares(z - 1)  # This is the chi-squared ball.

    if args.dataset == "CIFAR-10":
        data_folder = "datasets/CIFAR10"
    else:
        data_folder = "datasets/QMNIST"
    path_to_data = path_to_scratch + data_folder

    path_to_experiment_folder = os.path.join(
        path_to_scratch,
        "experiments/robust_cv", args.dataset.lower()"
    )
    
    if not os.path.exists(path_to_experiment_folder):
        os.mkdir(path_to_experiment_folder)
    path_to_experiment_files = os.path.join(path_to_experiment_folder,args.experiment_name)
    
    print("***** STARTING MULTICLASS EXPERIMENT WITH DATASET {} ****".format(args.dataset))

        
    if (args.dataset == "CIFAR-10") or (args.dataset == "QMNIST"):
        dataset_split_list = ["train", "val", "testV2"]
        
        print("*** Loading data ***")
        data_cifar = {}
        for dataset in dataset_split_list:
            data_cifar[dataset] = np.load(path_to_data + "/np_data/" + dataset + ".npy",
                                         allow_pickle=True).item()
            
        df_multitrial_summary = initialize_df_multitrial_summary(
            ["Marginal", "CQC"], 
            ["NaiveConformal", "RobustConformal"],
             args.nTrials)   
    
    print(df_multitrial_summary)
        
    index = 0
    for M in range(args.nTrials):
        print("*** TRIAL {} ****".format(M))
        K.clear_session()
        
        if (args.dataset == "CIFAR-10") or (args.dataset == "QMNIST"):
            # Randomly split between validation / calibration
            val_permutation_split, val_splitted_arrays = cf_utils.split_conformal(
                [data_cifar["val"]["features"], data_cifar["val"]["labels"],  data_cifar["val"]["scores"]],
                split_percentages=np.array([70, 30])
            )
            val_splitted_features, val_splitted_labels, val_splitted_scores = val_splitted_arrays
            
        
        data_features = dict(zip(["val", "calib"], val_splitted_features))
        data_labels = dict(zip(["val", "calib"], val_splitted_labels))
        data_scores = dict(zip(["val", "calib"], val_splitted_scores))
        
        for dataset in ["train", "testV2"]:
            data_features[dataset] = data_cifar[dataset]["features"]
            data_labels[dataset]   = data_cifar[dataset]["labels"]
            data_scores[dataset]   = data_cifar[dataset]["scores"]
        

        print("Training Set Size: {}".format(data_features["train"].shape[0]))
        print("Validation Set Size: {}".format(data_features["val"].shape[0]))
        print("Calibration Set Size: {}".format(data_features["calib"].shape[0]))
        print("Testing Set Size: {}".format(data_features["testV2"].shape[0]))
        
        # Run Conformal Methods
        cqc_confidence_sets, marginal_confidence_sets, coverage_cqc, coverage_marginal, data_scores_quantiles = run_cqc_experiment(
            data_features,
            data_labels,
            data_scores,
            alpha=alpha,
            data_for_standard_quantile_function="val",
            epochs_quantile=args.epochs_quantile,
            verbose=True
        )
        
        df_multitrial_summary["AvgSetSize"][index] = np.array(marginal_confidence_sets, dtype=bool).sum(axis=1).mean()
        df_multitrial_summary["Coverage"][index] = coverage_marginal
        index+=1
        
        df_multitrial_summary["AvgSetSize"][index] = np.array(cqc_confidence_sets, dtype=bool).sum(axis=1).mean()
        df_multitrial_summary["Coverage"][index] = coverage_cqc
        index+=1
        

        slab_quantiles = {"Marginal": np.zeros(args.n_slabs_directions), 
                          "CQC": np.zeros(args.n_slabs_directions)}
        q_slab = {}        
        
        S_calib_std = -(data_scores["calib"] * data_labels["calib"]).sum(axis=1)
        S_calid_cqc = S_calib_std + data_scores_quantiles["calib"]
        
        # Run robust validation with slabs
        if args.rho_validation == "slab_quantile":
            for slab_idx in range(args.n_slabs_directions):
                if slab_idx % 10 == 0:
                    print("Slab Id={}".format(slab_idx))
                # Might want to try out different sampling schemes for direction
                direction = np.random.randn(data_features["calib"].shape[1])
                direction = direction / np.linalg.norm(direction)

                slab_quantiles["Marginal"][slab_idx] = find_worst_case_slab_quantile(
                    direction, data_features["calib"], S_calib_std, alpha, args.delta_slab)

                slab_quantiles["CQC"][slab_idx] = find_worst_case_slab_quantile(
                    direction, data_features["calib"], S_calid_cqc, alpha, args.delta_slab)

            # Computing the new robust threshold
            for conformal_method in ["Marginal", "CQC"]:
                q_slab[conformal_method] = np.quantile(
                    slab_quantiles[conformal_method], 1 - args.alpha_slab, interpolation="higher")
                
        elif args.rho_validation == "learnable_direction_OLS":
            q_slab["Marginal"] = learnable_direction_quantile(
                    data_features["calib"],
                    S_calib_std,
                    np.arange(data_features["calib"].shape[0]),
                    model="OLS",
                    alpha=alpha,
                    delta=args.delta_slab
                )
            
            q_slab["CQC"] = learnable_direction_quantile(
                    data_features["calib"],
                    S_calid_cqc,
                    np.arange(data_features["calib"].shape[0]),
                    model="OLS",
                    alpha=alpha,
                    delta=args.delta_slab
                )
            
        elif args.rho_validation == "learnable_direction_SVM":
            q_slab["Marginal"] = learnable_direction_quantile(
                    data_features["calib"],
                    S_calib_std,
                    np.arange(data_features["calib"].shape[0]),
                    model="SVM",
                    alpha=alpha,
                    delta=args.delta_slab
            )
            
            q_slab["CQC"] = learnable_direction_quantile(
                    data_features["calib"],
                    S_calid_cqc,
                    np.arange(data_features["calib"].shape[0]),
                    model="SVM",
                    alpha=alpha,
                    delta=args.delta_slab
            )
                
        else:
            raise NotImplementedError("Rho Validation Method Not Implemented")
        
        
        marginal_robust_confidence_sets = (data_scores["testV2"] >= - q_slab["Marginal"])
        cqc_robust_confidence_sets = (data_scores["testV2"] >= data_scores_quantiles["testV2"][:,np.newaxis] - q_slab["CQC"])
            
        df_multitrial_summary["AvgSetSize"][index] = np.array(marginal_robust_confidence_sets, dtype=bool).sum(axis=1).mean()
        df_multitrial_summary["Coverage"][index] = (marginal_robust_confidence_sets * data_labels["testV2"]).sum(axis=1).mean()
        # What rho does it correspond to?
        df_multitrial_summary["Rho"][index] = find_rho_for_quantile(
            S_calib_std, q_slab["Marginal"], chisq, alpha=alpha, delta=args.delta_slab)
        index+= 1
        
        df_multitrial_summary["AvgSetSize"][index] = np.array(cqc_robust_confidence_sets, dtype=bool).sum(axis=1).mean()
        df_multitrial_summary["Coverage"][index] = (cqc_robust_confidence_sets * data_labels["testV2"]).sum(axis=1).mean()
        # What rho does it correspond to?
        df_multitrial_summary["Rho"][index] = find_rho_for_quantile(
            S_calid_cqc, q_slab["CQC"], chisq, alpha=alpha, delta=args.delta_slab)
        index+= 1
            
        
    print("*** Saving Results ***")

    df_multitrial_summary.to_csv(
        path_to_experiment_files + "-summary.csv")