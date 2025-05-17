from src.utils import *
import pandas as pd

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from brokenaxes import brokenaxes
from matplotlib.lines import Line2D

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



reg_results = pd.read_csv('../experiments/imgnet/robust_to_both/reg_result_hist_0.01_0.25_-0.25.csv').to_numpy()
coverage_results = [reg_results[:, i] for i in range(1, 8)]
size_results = [reg_results[:, j] for j in range(8, 15)]

plot_cp(coverage_results, plt_type='Coverage', plt_name=f'imgnet_{0.01}_{0.25}_{-0.25}_cover.png',
        save_dir='figures', )
plot_cp(size_results, plt_type='Size', plt_name=f'imgnet_{0.01}_{0.25}_{-0.25}_size.png', save_dir='figures', ylims=((0, 20), (885, 915)))


reg_results = pd.read_csv('../experiments/imgnet/robust_to_both/reg_result_hist_0.025_0.5_-0.5.csv').to_numpy()
coverage_results = [reg_results[:, i] for i in range(1, 8)]
size_results = [reg_results[:, j] for j in range(8, 15)]

plot_cp(coverage_results, plt_type='Coverage', plt_name=f'imgnet_{0.025}_{0.5}_{-0.5}_cover.png',
        save_dir='figures', )
plot_cp(size_results, plt_type='Size', plt_name=f'imgnet_{0.025}_{0.5}_{-0.5}_size.png', save_dir='figures', ylims=((0, 25), (885, 915)))


reg_results = pd.read_csv('../experiments/imgnet/robust_to_both/reg_result_hist_0.05_1.0_-1.0.csv').to_numpy()
coverage_results = [reg_results[:, i] for i in range(1, 8)]
size_results = [reg_results[:, j] for j in range(8, 15)]

plot_cp(coverage_results, plt_type='Coverage', plt_name=f'imgnet_{0.05}_{1.0}_{-1.0}_cover.png',
        save_dir='figures', )
plot_cp(size_results, plt_type='Size', plt_name=f'imgnet_{0.05}_{1.0}_{-1.0}_size.png', save_dir='figures', ylims=((0, 30), (885, 915)))


