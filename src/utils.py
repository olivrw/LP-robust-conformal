import os
import torch
import ot
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
from matplotlib.ticker import FixedLocator
from brokenaxes import brokenaxes
from matplotlib.lines import Line2D
import pandas as pd
import re
import matplotlib.colors as mcolors


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
        return -F.log_softmax(outputs, dim=1)    
    

def indicator_cost_plan(samples1, samples2, epsilon, reg=0.05):
    n = len(samples1)
    m = len(samples2)
    a = np.ones(n) / n  
    b = np.ones(m) / m  

    x = samples1[:, np.newaxis]
    y = samples2[np.newaxis, :]
    cost_matrix = (np.abs(x - y) >= epsilon).astype(float)
    
    # Compute optimal transport plan
    transport_plan = ot.sinkhorn(a, b, cost_matrix, reg)
    total_cost = np.sum(transport_plan * cost_matrix)
    return total_cost, transport_plan, cost_matrix


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


_ORDER_CACHE: list[int] | None = None

def _compute_order_by_mean(data_arr: list[np.ndarray]) -> list[int]:
    means = [float(np.nanmean(g)) for g in data_arr]
    return list(np.argsort(means))


def plot_cp(
    data,
    plt_type,
    plt_name,
    *,
    alpha: float = 0.1,
    save_dir: str | None = None,
    group_labels: list[str] | None = None,
    highlight_groups: tuple[str, ...] = (
        r"$\mathbf{LP}_{\boldsymbol{\epsilon}}$",
        r"$\mathbf{LP}^{\mathbf{est}}_{\boldsymbol{\epsilon}}$",
    ),
    ylims=((0, 30), (885, 915)),
    target_band: tuple[float, float] | None = (0.89, 0.91),
    highlight_color: str = "#d62728",  
    other_color: str = "#7393B3",      
):

    global _ORDER_CACHE

    if group_labels is None:
        group_labels = [
            "SC",
            r"$\mathbf{LP}_{\boldsymbol{\epsilon}}$",
            r"$\mathbf{LP}^{\mathbf{est}}_{\boldsymbol{\epsilon}}$",
            r"$\chi^2$",
            "FG‑CP",
            "RSCP",
            "Weight",
        ]

    if len(data) != len(group_labels):
        raise ValueError("*data* and *group_labels* lengths mismatch.")

    # Convert to ndarray for numeric operations
    data_arr = [np.asarray(g) for g in data]

    marker_map_spec = {
        r"$\mathbf{LP}_{\boldsymbol{\epsilon}}$": "^",  
        r"$\mathbf{LP}^{\mathbf{est}}_{\boldsymbol{\epsilon}}$": "v",  
        "SC": "o",          
        r"$\chi^2$": "s", 
        "FG‑CP": "D",       
        "RSCP": "H",        
        "Weight": "X",      
    }

    fallback_cycle = iter(["P", "<", ">", "*", "+"])
    marker_map: dict[str, str] = {}
    for lbl in group_labels:
        if lbl in marker_map_spec:
            marker_map[lbl] = marker_map_spec[lbl]
        else:
            marker_map[lbl] = next(fallback_cycle, "o")

    if plt_type == "Coverage":
        order = _compute_order_by_mean(data_arr)
        _ORDER_CACHE = order
    elif plt_type == "Size" and _ORDER_CACHE is not None:
        order = _ORDER_CACHE
    else:
        order = list(range(len(data_arr)))

    data_arr = [data_arr[i] for i in order]
    group_labels = [group_labels[i] for i in order]

    colours = [
        highlight_color if any(h in lbl for h in highlight_groups) else other_color
        for lbl in group_labels
    ]

    use_broken = plt_type == "Size" and ylims is not None
    fig_size = (6.0, 3.0)

    if use_broken:
        fig = plt.figure(figsize=fig_size)
        ax = brokenaxes(ylims=ylims, hspace=0.05, despine=False, fig=fig)
    else:
        fig, _ax = plt.subplots(figsize=fig_size)
        ax = _ax

    for i, (grp, lbl, col) in enumerate(zip(data_arr, group_labels, colours)):
        x_c = i
        jitter = (np.random.rand(len(grp)) - 0.5) * 0.35
        x_vals = x_c + jitter
        marker = marker_map[lbl]
        
        ax.scatter(
            x_vals,
            grp,
            marker=marker,
            color=col,
            alpha=0.65 if col == highlight_color else 0.9,
            edgecolor="white",
            linewidth=0.5,
            s=120 if col == highlight_color else 80,
            zorder=3,
        )

        mean_val = float(np.nanmean(grp))
        ax.hlines(
            y=mean_val,
            xmin=x_c - 0.35,
            xmax=x_c + 0.35,
            color=col,
            linewidth=3,
            zorder=4,
        )

    locs = np.arange(len(data_arr))
    axes_to_fix = ax.axs if hasattr(ax, "axs") else [ax]

    for a in axes_to_fix:
        a.xaxis.set_major_locator(FixedLocator(locs))
        a.set_xticklabels(group_labels, fontsize=11)

        if plt_type == "Coverage" and target_band is not None:
            a.axhspan(*target_band, color="#b0e0a8", alpha=0.25, zorder=0)

    if plt_type == "Coverage":
        ax.axhline(1 - alpha, color="k", linestyle="--", linewidth=1.2, zorder=2)

    ax.set_title(plt_type, fontsize=14, pad=8)

    plt.tight_layout()

    if use_broken:
        for h in ax.diag_handles:
            h.remove()
        ax.draw_diags()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, plt_name), dpi=300, bbox_inches="tight")

    plt.show()


def eps_rho_plot(arr,
                 result_ot,
                 plt_type='Coverage',
                 scatter_points=True,
                 levels=50,
                 style='darkgrid',
                 context='talk',
                 figsize=(8, 6),
                 point_size=70,
                 alpha=0.8,
                 highlight_val=0.9,
                 savefig_path='wilds_cover_plot.png'):


    sns.set_style(style)
    sns.set_context(context)

    palette = 'rocket' if plt_type == 'Coverage' else 'mako'
    cmap    = sns.color_palette(palette, as_cmap=True)

    # unpack
    val, x, y               = arr.T
    val_ot_raw, x_ot, y_ot  = result_ot.T          

    norm = mcolors.Normalize(vmin=val.min(), vmax=val.max())

    fig, ax = plt.subplots(figsize=figsize)

    triang = tri.Triangulation(x, y)
    contour_f = ax.tricontourf(triang, val, levels=levels,
                               cmap=cmap, norm=norm)

    if scatter_points:
        interpolator   = tri.LinearTriInterpolator(triang, val)
        val_ot_interp  = interpolator(x_ot, y_ot)
        valid          = ~np.isnan(val_ot_interp)

        ax.scatter(x_ot[valid], y_ot[valid],
                   c=val_ot_interp[valid].data,
                   cmap=cmap, norm=norm,
                   s=point_size,
                   edgecolors='white',
                   linewidths=0.8,
                   alpha=alpha)

    hc = ax.tricontour(triang, val, levels=[highlight_val],
                       colors='white', linewidths=2, linestyles='--')
    if plt_type == 'Coverage':
        label_xy = [(1.4, 0.024)]
        ax.clabel(hc, fmt={highlight_val: rf'$Cov={highlight_val}$'}, inline=True, manual=label_xy)

    idx_grid = np.argmin(np.abs(val - highlight_val))
    x_g, y_g, v_g = x[idx_grid], y[idx_grid], val[idx_grid]

    ax.scatter(x_g, y_g,
               s=point_size*1.8,
               facecolors='none', edgecolors='black',
               linewidths=3, zorder=4,
               label='Grid Search')

    ax.annotate(
        rf'$(\epsilon={x_g:.2f}, \rho={y_g:.3f})$'+'\n'+rf'$Size={v_g:.3f}$',
        xy=(x_g, y_g), xycoords='data',
        xytext=(-40, -30), textcoords='offset points',
        arrowprops=dict(arrowstyle='->', lw=1.5),
        fontsize=13, ha='right', va='bottom', color='white')

    if scatter_points and valid.any():
        idx_ot = np.argmin(np.abs(val_ot_interp[valid] - highlight_val))
        x_o = x_ot[valid][idx_ot]
        y_o = y_ot[valid][idx_ot]
        v_o = val_ot_raw[valid][idx_ot]

        ax.scatter(x_o, y_o,
                   s=point_size*1.8, marker='D',
                   facecolors='none', edgecolors='black',
                   linewidths=3, zorder=4,
                   label='Estimated')

        ax.annotate(
            rf'$(\epsilon={x_o:.2f}, \rho={y_o:.3f})$'+'\n'+rf'$Size={v_o:.3f}$',
            xy=(x_o, y_o), xycoords='data',
            xytext=(35, 45), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', lw=1.5),
            fontsize=13, ha='left', va='top', color='white')

    # legend
    ax.legend(loc='upper right', fontsize=14, frameon=True)

    fig.colorbar(contour_f, ax=ax)

    ax.set_xlabel(r'$\epsilon$', fontsize=25)
    ax.set_ylabel(r'$\rho$',    fontsize=25)
    ax.set_title(plt_type,      fontsize=25)
    plt.tight_layout()

    if savefig_path:
        fig.savefig(savefig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {savefig_path}")
        
    plt.show()

    return fig, ax


def parse_tensor(x):
    match = re.match(r"tensor\((.*)\)", str(x))
    if match:
        return float(match.group(1))
    return float(x)
