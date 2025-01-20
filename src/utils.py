import os
import torch
import numpy as np
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_img(image, labels):
    grid = torchvision.utils.make_grid(image, nrow=3)
    plt.figure(figsize=(7, 7))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print('labels: ', labels)


def load_imgnet_valdata(data_dir, preprocess, cal_ratio, batch_size, seed):
    # load dataset
    val_dataset = datasets.ImageFolder(root=data_dir, transform=preprocess)
    # split data and load
    cal_size = int(cal_ratio * len(val_dataset))
    test_size = len(val_dataset) - cal_size
    generator = torch.Generator().manual_seed(seed)
    cal_dataset, test_dataset = random_split(val_dataset, [cal_size, test_size], generator=generator)
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return cal_loader, test_loader


def load_mnist_valdata(data_dir, preprocess, cal_ratio, batch_size, seed):
    # load dataset
    val_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=preprocess)
    # split data and load
    cal_size = int(cal_ratio * len(val_dataset))
    test_size = len(val_dataset) - cal_size
    generator = torch.Generator().manual_seed(seed)
    cal_dataset, test_dataset = random_split(val_dataset, [cal_size, test_size], generator=generator)
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return cal_loader, test_loader


def nll_score(model, features, labels=None):
    with torch.no_grad():
        outputs = model(features.to(device))
        return -F.log_softmax(outputs, dim=1)    # shape (n_images, n_labels)


def perturb_test_data(features, labels, corrupt_ratio, noise_upper=1., noise_lower=-1., worst_case=False):
    n_ex = labels.shape[0]
    # add uniform noise
    if worst_case is True:
        max_noise_level = np.max(np.abs((noise_upper, noise_lower)))
        noise = torch.where(torch.rand_like(features) > 0.5, max_noise_level, -max_noise_level)
    else:
        noise = (noise_lower - noise_upper) * torch.rand(features.size()) + noise_upper
    noised_features = features + noise

    # corrupt labels
    if corrupt_ratio == 0.:
        perturbed_labels = labels
    else:    # roll the labels
        perturbed_labels = torch.clone(labels)
        pert_idx = np.random.choice(n_ex, int(corrupt_ratio * n_ex), replace=False)
        vals_perturb = perturbed_labels[pert_idx]
        vals_perturb = torch.roll(vals_perturb, 1)
        perturbed_labels[pert_idx] = vals_perturb
    return noised_features, perturbed_labels


def perturb_test_scores(tst_scores, corrupt_ratio, noise_upper=1., noise_lower=-1., worst_case=False):
    n_ex = tst_scores.shape[0]
    # corruption
    if corrupt_ratio == 0.:
        tst_scores = tst_scores
    else:
        perturbed_tstscores = torch.clone(tst_scores)
        pert_idx = np.random.choice(n_ex, int(corrupt_ratio * n_ex), replace=False)
        scores_perturb = perturbed_tstscores[pert_idx]
        scores_perturb = torch.roll(scores_perturb, 1)
        perturbed_tstscores[pert_idx] = scores_perturb

    # add uniform noise
    if worst_case is True:
        max_noise_level = np.max(np.abs((noise_upper, noise_lower)))
        noise = torch.where(torch.rand_like(tst_scores) > 0.5, max_noise_level, -max_noise_level)
    else:
        noise = (noise_lower - noise_upper) * torch.rand(tst_scores.size()) + noise_upper
    
    return perturbed_tstscores + noise


def plot_cp(data, plt_type, plt_name, alpha=0.1, 
            group_labels=['SC', '$LP_\epsilon$', '$\chi^2$']):
    
    colors = ['#1f77b4', '#dc143c', '#2ca02c']

    plt.figure(figsize=(3, 2.8))
    for i, group_data in enumerate(data):
        group_data = np.array(group_data)
        x_center = i
        jitter = (np.random.rand(len(group_data)) - 0.5) * 0.4  
        x_vals = x_center + jitter

        col = colors[i % len(colors)]

        # Scatter for each group
        plt.scatter(x_vals, group_data,
                    color=col,
                    alpha=0.5,
                    edgecolor='white',
                    s=100)

        # Draw a horizontal line for the mean
        mean_val = np.mean(group_data)
        plt.hlines(y=mean_val,
                   xmin=x_center - 0.4, 
                   xmax=x_center + 0.4,
                   color=col,
                   linewidth=3)

    plt.xticks(range(len(data)), group_labels, fontsize=11)
    plt.title(plt_type, fontsize=15)
    if plt_type == 'Coverage':
        plt.axhline(y=1 - alpha, color='darkred', linestyle='-', alpha=0.9, linewidth=2)

    plt.tight_layout()
    plt.savefig(plt_name + ".png", dpi=300, bbox_inches='tight')
    plt.show()

    
def eps_rho_plot(arr,
                 plt_type='Coverage',
                 scatter_points=True, 
                 levels=50, 
                 style='darkgrid',
                 context='talk',
                 figsize=(8, 6),
                 point_size=70,
                 alpha=0.8,
                 highlight_val=0.9, 
                 savefig_path=None):

    sns.set_style(style)
    sns.set_context(context)
    
    if plt_type == 'Coverage':
        palette = 'rocket'
    else:
        palette='mako'
    cmap = sns.color_palette(palette, as_cmap=True)

    val = arr[:, 0]
    x = arr[:, 1]
    y = arr[:, 2]

    triang = tri.Triangulation(x, y)

    fig, ax = plt.subplots(figsize=figsize)
    contour_f = ax.tricontourf(triang, val, levels=levels, cmap=cmap)

    if scatter_points:
        sns.scatterplot(x=x, y=y, hue=val, palette=palette, 
                        alpha=alpha, edgecolor='white', 
                        s=point_size, ax=ax, legend=False)

    highlight_level = [highlight_val]
    highlight_contours = ax.tricontour(triang, val, levels=highlight_level,
                                       colors='black', linewidths=2, linestyles='--')
    if plt_type == 'Coverage':
        ax.clabel(highlight_contours, inline=True, 
                  fmt={highlight_val: f'val={highlight_val}'})

    cbar = fig.colorbar(contour_f, ax=ax)
    cbar.set_label(plt_type)
    ax.set_xlabel(r'$\epsilon$', fontsize=25)
    ax.set_ylabel(r'$\rho$', fontsize=25)
    plt.tight_layout()

    if savefig_path is not None:
        fig.savefig(savefig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefig_path}")

    return fig, ax
