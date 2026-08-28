from pathlib import Path

import iplotx as ipx
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import seaborn as sns
from bowside.dataset import (
    PseudotimeDataset,
    PseudotimeTimepointWrapper,
    split_by_time_point,
)
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set up plotting parameters
sc.set_figure_params(vector_friendly=True, dpi_save=600, scanpy=False, fontsize=7)
scv.set_figure_params(vector_friendly=True, dpi_save=600, fontsize=7)
plt.style.use("../config/matplotlibrc")

FIG_DIR = Path("../figures/subfigures")
DATA_DIR = Path("../data")
INCH_PER_CM = 1 / 2.54

# Color palettes
# Palette for train/val/test split, Paul Tol's high contrast
# https://sronpersonalpages.nl/~pault/
SPLIT_PALETTE = ["#004488", "#DDAA33", "#BB5566"]

# Palette for trajectories: from Paul Tol's muted
TRAJECTORY_PALETTE = ["#117733", "#AA4499", "#88CCEE"]

# Palette for cell types, Paul Tol's bright
CELL_TYPE_PALETTE = [
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
]


def make_overview_subfigures():
    fig_dir = FIG_DIR / "method_overview"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Make expression matrices
    rng = np.random.default_rng(seed=42)
    dropout_rate = 0.5
    n_cells = 10
    n_genes = 15
    mat_size = (n_cells, n_genes)
    figsize = (1.2 * INCH_PER_CM, 1.2 * INCH_PER_CM)
    for i in range(5):
        exp_mat = rng.random(mat_size) * rng.choice(
            [0, 1], mat_size, p=[dropout_rate, 1 - dropout_rate]
        )
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        sns.heatmap(exp_mat, cmap="Greens", ax=ax, cbar=False)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(top=False, right=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        fig.savefig(fig_dir / f"exp_mat_{i}")
        plt.close(fig)

    # Make prior graph
    adj_mat = np.asarray(
        [
            [4, -3, 1, 2, 0, 0, 2],
            [-3, 4, 0, -1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 1.5],
            [0, 0, 0, 0, 0, 1.5, 0],
        ]
    )
    prior_graph = nx.from_numpy_array(
        adj_mat,
        nodelist=[f"g{i + 1:d}" for i in range(adj_mat.shape[0])],  # type: ignore
        create_using=nx.DiGraph,
    )
    pos = {
        "g1": [1, 1.5],
        "g2": [-0.2, 1.5],
        "g3": [1.8, 0.5],
        "g4": [0.5, -0.2],
        "g5": [-0.4, -1.2],
        "g6": [0.5, -2],
        "g7": [1.5, -0.8],
    }
    pos = {key: np.asarray(lst) for key, lst in pos.items()}
    figsize = (2 * INCH_PER_CM, 2 * INCH_PER_CM)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ipx.network(
        prior_graph,
        pos,  # type: ignore
        node_labels=False,
        style={
            "node": {
                "size": 4,
                "facecolor": "lightgrey",
            },
            "edge": {
                "shrink": 1,
                "linewidth": 0.5,
                "arc": True,
                "tension": 0.15,
                "looptension": 5,
                "arrow_marker": ")>",
                "arrow_width": 1.5,
                "arrow_height": 1.5,
            },
        },
        ax=ax,
    )
    sns.despine(left=True, bottom=True)
    fig.savefig(fig_dir / "prior_graph")
    plt.close(fig)

    # Make learned graph
    learned_graph: nx.DiGraph = prior_graph.copy()
    learned_graph.remove_edges_from([("g4", "g5"), ("g4", "g7")])

    # Setup colormap
    norm = mcolors.Normalize(vmin=-4, vmax=4)
    cmap = sns.cm.vlag
    nx.draw_networkx_nodes(learned_graph, pos, node_color="grey", node_size=10, ax=ax)

    # Style edges
    edge_colors = {}
    edge_arrowmarkers = []
    for u, v, d in learned_graph.edges(data=True):
        w = d["weight"]

        # Map weight to seaborn colormap
        edge_colors[(u, v)] = cmap(norm(w))

        # Different arrows for activation and inhibition
        if w > 0:
            edge_arrowmarkers.append(")>")  # standard arrow
        else:
            edge_arrowmarkers.append("|")  # blunt arrow

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ipx.network(
        learned_graph,
        pos,  # type: ignore
        node_labels=False,
        style={
            "node": {
                "size": 4,
                "facecolor": "lightgrey",
            },
            "edge": {
                "shrink": 1,
                "linewidth": 0.5,
                "arc": True,
                "tension": 0.15,
                "looptension": 5,
                "color": edge_colors,
                "arrow_marker": edge_arrowmarkers,
                "arrow_width": 1.5,
                "arrow_height": 1.5,
            },
        },
        ax=ax,
    )

    sns.despine(left=True, bottom=True)
    fig.savefig(fig_dir / "learned_graph")
    plt.close(fig)

    # Make KO graph
    ko_graph: nx.DiGraph = learned_graph.copy()
    edges_to_remove = []
    edge_arrowmarkers = []
    for u, v, d in ko_graph.edges(data=True):
        if u == "g1":
            edges_to_remove.append((u, v))
        else:
            # Different arrows for activation and inhibition
            if d["weight"] > 0:
                edge_arrowmarkers.append(")>")  # standard arrow
            else:
                edge_arrowmarkers.append("|")  # blunt arrow
    ko_graph.remove_edges_from(edges_to_remove)
    node_colors = {
        node: "lightgrey" if node != "g1" else "k" for node in ko_graph.nodes()
    }

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ipx.network(
        ko_graph,
        pos,  # type: ignore
        node_labels=False,
        style={
            "node": {
                "size": 4,
                "facecolor": node_colors,
            },
            "edge": {
                "shrink": 1,
                "linewidth": 0.5,
                "arc": True,
                "tension": 0.15,
                "looptension": 5,
                "color": edge_colors,
                "arrow_marker": edge_arrowmarkers,
                "arrow_width": 1.5,
                "arrow_height": 1.5,
            },
        },
        ax=ax,
    )

    sns.despine(left=True, bottom=True)
    fig.savefig(fig_dir / "ko_graph")
    plt.close(fig)

    # Data and simulations
    arrays = np.load("../output/hematopoiesis/bowside_results_overview.npz")
    figsize = (2.0 * INCH_PER_CM, 2.0 * INCH_PER_CM)
    for i in range(arrays["data"].shape[-1]):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        arr = arrays["data"][:, :, i]
        sns.violinplot(
            arr,
            color="grey",
            inner=None,
            linewidth=0.25,
            cut=0,
            density_norm="width",
            ax=ax,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t", labelpad=1.0, fontsize=5)
        ax.set_ylabel(f"$g_{i + 1}$", labelpad=0.05, fontsize=5)
        sns.despine(top=False, right=False)
        fig.savefig(fig_dir / f"data_g{i + 1}")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        arr = arrays["simulations"][:, :, i]
        ax.plot(arr, alpha=0.1, c="k", lw=0.25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("t", labelpad=1.0, fontsize=5)
        ax.set_ylabel(f"$g_{i + 1}$", labelpad=0.05, fontsize=5)
        sns.despine(top=False, right=False)
        fig.savefig(fig_dir / f"simulation_{i + 1}")
        plt.close(fig)

    # Inferred dynamics
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for i in range(arrays["simulations"].shape[1]):
        points = arrays["simulations"][:, i, [0, 1]]
        segments = np.stack([points[:-1], points[1:]], axis=1).tolist()
        line = LineCollection(
            segments,
            cmap="viridis_r",
            linewidth=0.25,
            alpha=0.5,
        )
        line.set_array(np.linspace(0.0, 1.0, len(segments)))
        ax.add_collection(line)
    cax = inset_axes(ax, width="5%", height="25%", loc="upper right", borderpad=1.5)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=mcolors.Normalize(0.0, 1.0), cmap="viridis_r"),
        cax=cax,
    )
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.25)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels([r"$t_0$", "T"])
    cbar.ax.tick_params(length=0, pad=1.0)
    ax.set_xticks([])
    ax.set_xlabel("$g_1$", labelpad=2.0, fontsize=5)
    ax.set_xlim(-0.03, 0.7)
    ax.set_yticks([])
    ax.set_ylabel("$g_2$", labelpad=0.5, fontsize=5)
    ax.set_ylim(-0.03, 1.05)
    sns.despine(top=False, right=False)
    fig.savefig(fig_dir / "inferred_dynamics")
    plt.close(fig)

    # Trajectory inference
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for i in range(arrays["trajectories"].shape[0]):
        cluster_simulations = arrays["trajectory_labels"] == i + 1
        ax.plot(
            np.arange(arrays["simulations"].shape[0]),
            arrays["simulations"][:, cluster_simulations, 0],
            c=TRAJECTORY_PALETTE[i],
            alpha=0.1,
            lw=0.25,
        )
        ax.plot(
            np.arange(arrays["simulations"].shape[0]),
            arrays["trajectories"][i, :, 0],
            c=TRAJECTORY_PALETTE[i],
            lw=0.5,
        )
    ax.set_xticks([])
    ax.set_xlabel("t", labelpad=2.0, fontsize=5)
    ax.set_yticks([])
    ax.set_ylabel("$g_1$", labelpad=0.5, fontsize=5)
    sns.despine(top=False, right=False)
    fig.savefig(fig_dir / "trajectory_inference")
    plt.close(fig)


def make_hematopoiesis_subfigures():
    fig_dir = FIG_DIR / "hematopoiesis"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load AnnData object
    adata = sc.read_h5ad(DATA_DIR / "hematopoiesis_bowside.h5ad")
    embeddings_df = sc.get.obs_df(
        adata,
        keys=["cell_type", "palantir_pseudotime", "bowside_fate_entropy"],
        obsm_keys=[("X_draw_graph_fa", 0), ("X_draw_graph_fa", 1)],
    ).rename(columns={"X_draw_graph_fa-0": "FA1", "X_draw_graph_fa-1": "FA2"})
    embeddings_df[["bowside_erythroid_probability", "bowside_myeloid_probability"]] = (
        adata.obsm["bowside_fate_probabilities"].copy()
    )

    # Plot embeddings
    figsize = (4.5 * INCH_PER_CM, 5.5 * INCH_PER_CM)
    fig, axs = plt.subplots(
        2,
        2,
        figsize=figsize,
        width_ratios=[5, 2],
        layout="constrained",
    )
    sns.scatterplot(
        embeddings_df,
        x="FA1",
        y="FA2",
        hue="cell_type",
        palette=adata.uns["cell_type_colors"],
        linewidth=0,
        s=2,
        rasterized=True,
        ax=axs[0, 0],
    )
    handles, labels = axs[0, 0].get_legend_handles_labels()
    labels = [lab.replace("_", " ") for lab in labels]
    axs[0, 0].get_legend().remove()
    axs[0, 1].legend(
        handles,
        labels,
        title="cell type",
        alignment="left",
        loc="center",
        markerscale=2,
    )
    sm = plt.cm.ScalarMappable(
        cmap="viridis_r", norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
    )
    sns.scatterplot(
        embeddings_df,
        x="FA1",
        y="FA2",
        hue="palantir_pseudotime",
        hue_norm=sm.norm,
        palette="viridis_r",
        linewidth=0,
        s=2,
        rasterized=True,
        ax=axs[1, 0],
    )
    axs[1, 0].get_legend().remove()
    cax = inset_axes(
        axs[1, 1], width="15%", height="75%", loc="center left", borderpad=0
    )
    plt.colorbar(
        sm,
        cax=cax,
        label="pseudotime",
    )
    for ax in axs.flat:
        ax.set_aspect(1.0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
    sns.despine(fig, left=True, bottom=True)
    fig.savefig(fig_dir / "embeddings")
    plt.close(fig)

    # Plot data and simulations for selected genes
    dataset = PseudotimeDataset(
        adata,
        layer="MAGIC_imputed_data",
        mask_var="hematopoiesis_gene",
        pseudotime_key="palantir_pseudotime",
    )
    train_idx, val_idx, test_idx = split_by_time_point(  # type: ignore
        dataset, train_proportion=0.7, validation_proportion=0.1, test_proportion=0.2
    )
    train_set = PseudotimeTimepointWrapper(dataset, train_idx)
    val_set = PseudotimeTimepointWrapper(dataset, val_idx)
    test_set = PseudotimeTimepointWrapper(dataset, test_idx)
    gene_names = pd.Index(["Cebpa", "Cebpe", "Gata1", "Spi1"])
    gene_idx = dataset.gene_names.get_indexer(gene_names)
    figsize = (5.5 * INCH_PER_CM, 5.5 * INCH_PER_CM)
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, layout="constrained")
    alpha = 0.01
    for idx, ax in zip(gene_idx, axs.flat):
        for i, (set_name, data_subset) in enumerate(
            zip(["train", "validation", "test"], [train_set, val_set, test_set])
        ):
            y = data_subset[:].cpu()
            ax.scatter(
                data_subset.time_points.cpu().repeat(y.shape[0], 1),
                y[:, :, idx],
                color=SPLIT_PALETTE[i],
                linewidth=0,
                s=3,
                alpha=alpha,
                label=set_name,
                rasterized=True,
            )
        ax.plot(
            dataset.time_points.cpu(),
            adata.uns["bowside"]["simulations"][:, :, idx],
            color="k",
            alpha=alpha,
            linewidth=0.5,
            label="simulations",
            rasterized=True,
        )
        ax.set_xlabel("pseudotime", labelpad=1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylabel(dataset.gene_names[idx], labelpad=1.0)  # type: ignore
    fig.legend(
        handles=[
            plt.Line2D(  # type: ignore
                [], [], linewidth=0, marker="o", markersize=2, c=SPLIT_PALETTE[0]
            ),
            plt.Line2D(  # type: ignore
                [], [], linewidth=0, marker="o", markersize=2, c=SPLIT_PALETTE[1]
            ),
            plt.Line2D(  # type: ignore
                [], [], linewidth=0, marker="o", markersize=2, c=SPLIT_PALETTE[2]
            ),
            plt.Line2D([], [], linewidth=1.0, c="k"),  # type: ignore
        ],
        labels=["train", "validation", "test", "simulation"],
        loc="outside lower center",
        ncol=4,
        columnspacing=1.0,
    )
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.0)  # type: ignore
    fig.align_ylabels()
    fig.savefig(fig_dir / "data_and_simulations")
    plt.close(fig)

    # Trajectory inference
    figsize = (7 * INCH_PER_CM, 2.5 * INCH_PER_CM)
    fig, axs = plt.subplots(1, 4, figsize=figsize, layout="constrained")
    alpha = 0.01
    for idx, ax in zip(gene_idx, axs.flat):
        for i in range(adata.uns["bowside"]["trajectories"].shape[0]):
            cluster_simulations = (
                adata.uns["bowside"]["trajectory_inference_clusters"] == i + 1
            )
            ax.plot(
                dataset.time_points.cpu(),
                adata.uns["bowside"]["simulations"][:, cluster_simulations, idx],
                color=TRAJECTORY_PALETTE[i],
                alpha=alpha,
                linewidth=0.5,
                rasterized=True,
            )
            ax.plot(
                dataset.time_points.cpu(),
                adata.uns["bowside"]["trajectories"][i, :, idx],
                c=TRAJECTORY_PALETTE[i],
                lw=0.5,
            )
        ax.set_xlabel("pseudotime", labelpad=1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylabel(dataset.gene_names[idx], labelpad=1.0)  # type: ignore
    fig.legend(
        handles=[
            plt.Line2D(  # type: ignore
                [], [], linewidth=1.0, c=TRAJECTORY_PALETTE[0]
            ),
            plt.Line2D(  # type: ignore
                [], [], linewidth=1.0, c=TRAJECTORY_PALETTE[1]
            ),
        ],
        labels=["erythroid", "myeloid"],
        loc="outside lower center",
        ncol=2,
        columnspacing=1.0,
        borderpad=0.0,
    )
    fig.get_layout_engine().set(w_pad=0.0, wspace=0.0, rect=(0.01, 0.01, 0.98, 0.98))  # type: ignore
    fig.savefig(fig_dir / "trajectory_inference_genes")
    plt.close()

    figsize = (7 * INCH_PER_CM, 3 * INCH_PER_CM)
    fig, axs = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    sm = plt.cm.ScalarMappable(
        cmap="viridis_r", norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
    )
    cbar_kwargs = {"location": "bottom", "shrink": 0.4, "aspect": 12, "pad": 0.0}

    sns.scatterplot(
        embeddings_df,
        x="FA1",
        y="FA2",
        hue="bowside_erythroid_probability",
        hue_norm=sm.norm,
        palette="viridis_r",
        linewidth=0,
        s=2,
        rasterized=True,
        ax=axs[0],
    )
    axs[0].set_title("erythroid fate")
    cbar = plt.colorbar(sm, ax=axs[0], **cbar_kwargs)
    cbar.set_label("probability", labelpad=0)
    cbar.set_ticks([0, 0.5, 1.0])

    sns.scatterplot(
        embeddings_df,
        x="FA1",
        y="FA2",
        hue="bowside_myeloid_probability",
        hue_norm=sm.norm,
        palette="viridis_r",
        linewidth=0,
        s=2,
        rasterized=True,
        ax=axs[1],
    )
    axs[1].set_title("myeloid fate")
    cbar = plt.colorbar(sm, ax=axs[1], **cbar_kwargs)
    cbar.set_label("probability", labelpad=0)
    cbar.set_ticks([0, 0.5, 1.0])

    sm = plt.cm.ScalarMappable(
        cmap="viridis_r", norm=mcolors.Normalize(vmin=0.0, vmax=-np.log(0.5))
    )
    sns.scatterplot(
        embeddings_df,
        x="FA1",
        y="FA2",
        hue="bowside_fate_entropy",
        hue_norm=sm.norm,
        palette="viridis_r",
        linewidth=0,
        s=2,
        rasterized=True,
        ax=axs[2],
    )
    axs[2].set_title("fate entropy")
    cbar = plt.colorbar(sm, ax=axs[2], **cbar_kwargs)
    cbar.set_label("entropy", labelpad=0)
    cbar.set_ticks([0.0, -np.log(0.5)], labels=[f"{0:.1f}", f"{-np.log(0.5):.2f}"])

    for ax in axs:
        ax.get_legend().remove()
        ax.set_aspect(1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    sns.despine(left=True, bottom=True)
    fig.savefig(fig_dir / "trajectory_inference_embedding")
    plt.close()

    # Plot adjacency matrix
    adj_mat = pd.read_csv(
        "../output/hematopoiesis/hematopoiesis_adj_bowside.csv", index_col=0
    )
    grn_bowside = nx.from_pandas_adjacency(adj_mat, create_using=nx.DiGraph)

    adj_tflink = pd.read_csv(
        "../output/hematopoiesis/hematopoiesis_adj_list_tflink.csv"
    )
    grn_tflink = nx.from_pandas_edgelist(
        adj_tflink,
        source="source",
        target="target",
        edge_attr=True,
        create_using=nx.DiGraph,
    )

    # Remove edges below cutoff
    n_edges = 25
    sorted_weights = np.sort(np.abs(adj_mat.values), axis=None)
    edge_cutoff = sorted_weights[-n_edges]
    edges_to_remove = [
        (u, v)
        for u, v, d in grn_bowside.edges(data=True)
        if abs(d["weight"]) < edge_cutoff
    ]
    grn_bowside.remove_edges_from(edges_to_remove)
    pos = nx.circular_layout(grn_bowside)

    # Setup colormap
    vmax = max(abs(adj_mat.values.min()), adj_mat.values.max())
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = sns.cm.vlag

    # Style edges
    edge_linestyles = {}
    edge_colors = {}
    edge_arrowmarkers = []

    for u, v, d in grn_bowside.edges(data=True):
        w = d["weight"]

        # Style depending on presence in TFLink database
        if grn_tflink.has_edge(u, v):
            small_scale = grn_tflink.get_edge_data(u, v)["small_scale"]
            if small_scale == "Yes":
                edge_linestyles[(u, v)] = "-"
            else:
                edge_linestyles[(u, v)] = "--"
        else:
            edge_linestyles[(u, v)] = ":"

        # Map weight to seaborn colormap
        edge_colors[(u, v)] = cmap(norm(w))

        # Different arrows for activation and inhibition
        if w > 0:
            edge_arrowmarkers.append(")>")  # standard arrow
        else:
            edge_arrowmarkers.append("|")  # blunt arrow

    # Plot the graph
    fig, ax = plt.subplots(
        figsize=(5 * INCH_PER_CM, 5.5 * INCH_PER_CM), layout="constrained"
    )
    ipx.network(
        grn_bowside,
        pos,  # type: ignore
        node_labels=True,
        style={
            "node": {
                "size": 16,
                "facecolor": "lightgrey",
                "label_color": "k",
                "label_fontsize": 5,
                "label_fontfamily": "Arial",
            },
            "edge": {
                "shrink": 1,
                "linewidth": 0.75,
                "linestyle": edge_linestyles,
                "arc": True,
                "tension": 0.1,
                "looptension": 3,
                "color": edge_colors,
                "arrow_marker": edge_arrowmarkers,
                "arrow_width": 3,
                "arrow_height": 3,
            },
        },
        aspect=1,
        ax=ax,
    )
    cax = ax.inset_axes((0.9, 0.05, 0.03, 0.15), transform=ax.transAxes)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("weight", labelpad=1.0)
    handles = [
        plt.Line2D([], [], linewidth=1.0, linestyle=":", c="k"),  # type: ignore
        plt.Line2D([], [], linewidth=1.0, linestyle="--", c="k"),  # type: ignore
        plt.Line2D([], [], linewidth=1.0, linestyle="-", c="k"),  # type: ignore
    ]
    labels = ["none", "large-scale", "small-scale"]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        handlelength=2,
        columnspacing=1.0,
        ncol=3,
        title="TFLink evidence",
    )

    sns.despine(fig, left=True, bottom=True)
    fig.savefig(fig_dir / "bowside_grn_graph")
    plt.close()

    # Plot bistable switch
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(5 * INCH_PER_CM, 1 * INCH_PER_CM),
        layout="constrained",
        width_ratios=[30, 1],
    )
    bistable_switch = grn_bowside.subgraph(["Gata1", "Spi1"])
    edge_arrowmarkers = []
    for _, _, d in bistable_switch.edges(data=True):
        # Different arrows for activation and inhibition
        if d["weight"] > 0:
            edge_arrowmarkers.append(")>")  # standard arrow
        else:
            edge_arrowmarkers.append("|")  # blunt arrow

    ipx.network(
        bistable_switch,
        {"Gata1": [-0.3, 0.0], "Spi1": [0.3, 0.0]},
        node_labels=True,
        style={
            "node": {
                "size": 16,
                "facecolor": "lightgrey",
                "label_color": "k",
                "label_fontsize": 5,
                "label_fontfamily": "Arial",
            },
            "edge": {
                "shrink": 1,
                "linewidth": 0.75,
                "arc": True,
                "tension": 0.1,
                "looptension": 3,
                "color": edge_colors,
                "arrow_marker": edge_arrowmarkers,
                "arrow_width": 3,
                "arrow_height": 3,
            },
        },
        ax=axs[0],
    )
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=axs[1])
    cbar.set_label("weight", labelpad=1.0)
    fig.savefig(fig_dir / "bistable_switch")
    plt.close()

    # Plot adjacency matrix
    fig, ax = plt.subplots(
        figsize=(5 * INCH_PER_CM, 4.5 * INCH_PER_CM), layout="constrained"
    )
    annot = np.vectorize(lambda x: f"{x:.1f}")(adj_mat.values)
    annot[(annot == "0.0") | (annot == "-0.0")] = ""
    hm = sns.heatmap(
        adj_mat,
        cmap="vlag",
        center=0,
        cbar_kws={
            "shrink": 0.4,
            "aspect": 10,
            "ticks": [-6, -3, 0, 3],
        },
        annot=annot,
        annot_kws={"fontsize": 5},
        fmt="s",
        square=True,
        ax=ax,
    )
    hm.collections[0].colorbar.set_label("weight", labelpad=1.5)  # type: ignore
    sns.despine(fig, top=False, right=False)
    fig.savefig(fig_dir / "bowside_adj_mat")
    plt.close()

    # Plot KO simulations
    cluster_labels_ko = np.load("../output/hematopoiesis/cluster_labels_ko.npz")
    figsize = (7 * INCH_PER_CM, 5.5 * INCH_PER_CM)
    fig = plt.figure(figsize=figsize, layout="constrained")
    sub_figs = fig.subfigures(2, 1)
    alpha = 0.01

    ymaxs = [0.55, 1.2, 1.1, 0.7]
    yticks = [[0.0, 0.2, 0.4], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.3, 0.6]]
    axs_top = sub_figs[0].subplots(1, 4)
    for col, (idx, ax) in enumerate(zip(gene_idx, axs_top.flat)):
        for i in range(adata.uns["bowside"]["trajectories"].shape[0]):
            cluster_simulations = cluster_labels_ko["Gata1"] == i
            ax.plot(
                dataset.time_points.cpu(),
                adata.uns["bowside"]["simulations_ko_Gata1"][
                    :, cluster_simulations, idx
                ],
                color=TRAJECTORY_PALETTE[i],
                alpha=alpha,
                linewidth=0.5,
                rasterized=True,
            )
            ax.plot(
                dataset.time_points.cpu(),
                adata.uns["bowside"]["trajectories"][i, :, idx],
                c=TRAJECTORY_PALETTE[i],
                lw=0.5,
                ls="--",
            )
        ax.set_xlabel("pseudotime", labelpad=1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylabel(dataset.gene_names[idx], labelpad=1.0)  # type: ignore
        ax.set_ylim(None, ymaxs[col])
        ax.set_yticks(yticks[col])
    sub_figs[0].suptitle("Gata1 knockout")

    axs_bottom = sub_figs[1].subplots(1, 4)
    for col, (idx, ax) in enumerate(zip(gene_idx, axs_bottom.flat)):
        for i in range(adata.uns["bowside"]["trajectories"].shape[0]):
            cluster_simulations = cluster_labels_ko["Spi1"] == i
            ax.plot(
                dataset.time_points.cpu(),
                adata.uns["bowside"]["simulations_ko_Spi1"][
                    :, cluster_simulations, idx
                ],
                color=TRAJECTORY_PALETTE[i],
                alpha=alpha,
                linewidth=0.5,
                rasterized=True,
            )
            ax.plot(
                dataset.time_points.cpu(),
                adata.uns["bowside"]["trajectories"][i, :, idx],
                c=TRAJECTORY_PALETTE[i],
                lw=0.5,
                ls="--",
            )
        ax.set_xlabel("pseudotime", labelpad=1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_ylabel(dataset.gene_names[idx], labelpad=1.0)  # type: ignore
        ax.set_ylim(None, ymaxs[col])
        ax.set_yticks(yticks[col])
    sub_figs[1].suptitle("Spi1 knockout")
    fig.legend(
        handles=[
            plt.Line2D(  # type: ignore
                [], [], linewidth=1.0, c=TRAJECTORY_PALETTE[0]
            ),
            plt.Line2D(  # type: ignore
                [], [], linewidth=1.0, linestyle="--", c="k"
            ),
            plt.Line2D(  # type: ignore
                [], [], linewidth=1.0, c=TRAJECTORY_PALETTE[1]
            ),
            plt.Line2D(  # type: ignore
                [], [], linewidth=1.0, linestyle="-", c="k"
            ),
        ],
        labels=["erythroid", "WT trajectories", "myeloid", "KO simulations"],
        loc="outside lower center",
        ncol=2,
        columnspacing=1.0,
        handlelength=2,
        borderpad=0.0,
    )
    fig.get_layout_engine().set(w_pad=0.0, wspace=0.0, rect=(0.01, 0.01, 0.98, 0.98))  # type: ignore
    fig.savefig(fig_dir / "ko_simulations_Gata1_Spi1")
    plt.close()


def make_ipsc_subfigures():
    fig_dir = FIG_DIR / "ipscs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load AnnData object
    adata = sc.read_h5ad(DATA_DIR / "ipscs_bowside.h5ad")
    embeddings_df = sc.get.obs_df(
        adata,
        keys=["cell_set", "day", "bowside_fate_entropy"],
        obsm_keys=[("X_fle", 0), ("X_fle", 1)],
    ).rename(columns={"X_fle-0": "FLE1", "X_fle-1": "FLE2"})
    embeddings_df[
        [
            "bowside_stromal_probability",
            "bowside_ipsc_probability",
            "bowside_tropho_probability",
        ]
    ] = adata.obsm["bowside_fate_probabilities"].copy()

    # Plot embeddings
    figsize = (7.0 * INCH_PER_CM, 4.5 * INCH_PER_CM)
    fig, axs = plt.subplots(
        2,
        2,
        figsize=figsize,
        height_ratios=[3, 1],
        layout="constrained",
        gridspec_kw={"hspace": 0},
    )
    sns.scatterplot(
        embeddings_df,
        x="FLE1",
        y="FLE2",
        hue="cell_set",
        palette=CELL_TYPE_PALETTE,
        linewidth=0,
        s=0.25,
        rasterized=True,
        ax=axs[0, 0],
    )
    handles, labels = axs[0, 0].get_legend_handles_labels()
    labels = [
        lab.lower()
        if lab not in ["IPS", "MEF"]
        else {"IPS": "IPS", "MEF": "MEF/other"}[lab]
        for lab in labels
    ]
    axs[0, 0].get_legend().remove()
    axs[1, 0].legend(
        handles,
        labels,
        title="cell type",
        alignment="left",
        loc="upper center",
        ncol=2,
        columnspacing=1.0,
        markerscale=6,
        bbox_to_anchor=(0.5, 1.0),
    )
    sm = plt.cm.ScalarMappable(
        cmap="viridis_r", norm=mcolors.Normalize(vmin=0.0, vmax=18.0)
    )
    sns.scatterplot(
        embeddings_df,
        x="FLE1",
        y="FLE2",
        hue="day",
        hue_norm=sm.norm,
        palette="viridis_r",
        linewidth=0,
        s=0.25,
        rasterized=True,
        ax=axs[0, 1],
    )
    axs[0, 1].get_legend().remove()
    cax = axs[1, 1].inset_axes([0.25, 0.4, 0.5, 0.2])
    cbar = plt.colorbar(sm, cax=cax, pad=0.0, orientation="horizontal")
    cax.set_xticks([0, 6, 12, 18])
    cbar.set_label("day", labelpad=0)
    for ax in axs[0, :]:
        ax.set_aspect(1.0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
    sns.despine(fig, left=True, bottom=True)
    for ax in axs[1, :]:
        ax.set_axis_off()
    fig.savefig(fig_dir / "embeddings")
    plt.close()

    # Plot data and simulations for selected genes
    dataset = PseudotimeDataset(
        adata,
        layer="MAGIC_imputed_data",
        mask_var="reprogramming_gene",
        pseudotime_key="day",
        time_points=sorted(adata.obs["day"].unique()),
    )
    train_idx, val_idx, test_idx = split_by_time_point(  # type: ignore
        dataset, train_proportion=0.78, validation_proportion=0.11, test_proportion=0.11
    )
    train_set = PseudotimeTimepointWrapper(dataset, train_idx)
    val_set = PseudotimeTimepointWrapper(dataset, val_idx)
    test_set = PseudotimeTimepointWrapper(dataset, test_idx)
    gene_names = pd.Index(["Nanog", "Gata3", "Prrx1", "Tdgf1", "Msx2", "Csf1"])
    gene_idx = dataset.gene_names.get_indexer(gene_names)
    figsize = (10.5 * INCH_PER_CM, 4.5 * INCH_PER_CM)
    fig, axs = plt.subplots(2, 3, figsize=figsize, sharex=True, layout="constrained")
    alpha = 0.01
    for idx, ax in zip(gene_idx, axs.flat):
        for i, (set_name, data_subset) in enumerate(
            zip(["train", "validation", "test"], [train_set, val_set, test_set])
        ):
            y = data_subset[:].cpu()
            ax.scatter(
                data_subset.time_points.cpu().repeat(y.shape[0], 1),
                y[:, :, idx],
                color=SPLIT_PALETTE[i],
                linewidth=0,
                s=3,
                alpha=alpha,
                label=set_name,
                rasterized=True,
            )
        ax.plot(
            dataset.time_points.cpu(),
            adata.uns["bowside"]["simulations"][:, :, idx],
            color="k",
            alpha=alpha,
            linewidth=0.5,
            label="simulations",
            rasterized=True,
        )
        ax.set_xlabel("day", labelpad=1.0)
        ax.set_xticks([0, 6, 12, 18])
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.set_ylabel(dataset.gene_names[idx], labelpad=1.0)  # type: ignore
    for ax, title in zip(axs[0, :], ["iPSC", "trophoblast", "stromal"]):
        ax.set_title(title, fontsize=5, pad=2)
    fig.legend(
        handles=[
            plt.Line2D(  # type: ignore
                [], [], linewidth=0, marker="o", markersize=2, c=SPLIT_PALETTE[0]
            ),
            plt.Line2D(  # type: ignore
                [], [], linewidth=0, marker="o", markersize=2, c=SPLIT_PALETTE[1]
            ),
            plt.Line2D(  # type: ignore
                [], [], linewidth=0, marker="o", markersize=2, c=SPLIT_PALETTE[2]
            ),
            plt.Line2D([], [], linewidth=1.0, c="k"),  # type: ignore
        ],
        labels=["train", "validation", "test", "simulation"],
        loc="outside center right",
    )
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.0)  # type: ignore
    fig.align_ylabels()
    fig.savefig(fig_dir / "data_and_simulations")
    plt.close()

    # Trajectory inference
    figsize = (10.5 * INCH_PER_CM, 5 * INCH_PER_CM)
    fig, axs = plt.subplots(2, 3, figsize=figsize, layout="constrained")
    alpha = 0.01
    for idx, ax in zip(gene_idx, axs.flat):
        for i in range(adata.uns["bowside"]["trajectories"].shape[0]):
            cluster_simulations = (
                adata.uns["bowside"]["trajectory_inference_clusters"] == i + 1
            )
            ax.plot(
                dataset.time_points.cpu(),
                adata.uns["bowside"]["simulations"][:, cluster_simulations, idx],
                color=TRAJECTORY_PALETTE[i],
                alpha=alpha,
                linewidth=0.5,
                rasterized=True,
            )
            ax.plot(
                dataset.time_points.cpu(),
                adata.uns["bowside"]["trajectories"][i, :, idx],
                c=TRAJECTORY_PALETTE[i],
                lw=0.5,
            )
        ax.set_xlabel("day", labelpad=1.0)
        ax.set_xticks([0, 6, 12, 18])
        ax.set_ylabel(dataset.gene_names[idx], labelpad=1.0)  # type: ignore
    for ax, title in zip(axs[0, :], ["ipsc", "trophoblast", "stromal"]):
        ax.set_title(title, fontsize=5, pad=2)
    fig.legend(
        handles=[
            plt.Line2D(  # type: ignore
                [], [], linewidth=1.0, c=TRAJECTORY_PALETTE[1]
            ),
            plt.Line2D(  # type: ignore
                [], [], linewidth=1.0, c=TRAJECTORY_PALETTE[2]
            ),
            plt.Line2D(  # type: ignore
                [], [], linewidth=1.0, c=TRAJECTORY_PALETTE[0]
            ),
        ],
        labels=["ipsc", "trophoblast", "stromal"],
        loc="outside center right",
        ncol=1,
        borderpad=0.0,
    )
    fig.savefig(fig_dir / "trajectory_inference_genes")
    plt.close()

    figsize = (7 * INCH_PER_CM, 5 * INCH_PER_CM)
    fig, axs = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    sm = plt.cm.ScalarMappable(
        cmap="viridis_r", norm=mcolors.Normalize(vmin=0.0, vmax=1.0)
    )
    cbar_kwargs = {"location": "right", "shrink": 0.4, "aspect": 8, "pad": 0.0}

    for i, (name, title) in enumerate(
        zip(["ipsc", "tropho", "stromal"], ["ipsc", "trophoblast", "stromal"])
    ):
        ax = axs.flat[i]
        sns.scatterplot(
            embeddings_df,
            x="FLE1",
            y="FLE2",
            hue=f"bowside_{name}_probability",
            hue_norm=sm.norm,
            palette="viridis_r",
            linewidth=0,
            s=0.25,
            rasterized=True,
            ax=ax,
        )
        ax.set_title(f"{title} fate")
        cbar = plt.colorbar(sm, ax=ax, **cbar_kwargs)
        cbar.set_label("probability", labelpad=2)
        cbar.set_ticks([0, 0.5, 1.0])

    sm = plt.cm.ScalarMappable(
        cmap="viridis_r", norm=mcolors.Normalize(vmin=0.0, vmax=-np.log(1 / 3))
    )
    sns.scatterplot(
        embeddings_df,
        x="FLE1",
        y="FLE2",
        hue="bowside_fate_entropy",
        hue_norm=sm.norm,
        palette="viridis_r",
        linewidth=0,
        s=0.25,
        rasterized=True,
        ax=axs.flat[-1],
    )
    axs.flat[-1].set_title("fate entropy")
    cbar = plt.colorbar(sm, ax=axs.flat[-1], **cbar_kwargs)
    cbar.set_label("entropy", labelpad=0)
    cbar.set_ticks(
        [0.0, 0.5, -np.log(1 / 3)],
        labels=[f"{0:.1f}", f"{0.5:.1f}", f"{-np.log(1 / 3):.2f}"],
    )

    for ax in axs.flat:
        ax.get_legend().remove()
        ax.set_aspect(1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    sns.despine(left=True, bottom=True)
    fig.savefig(fig_dir / "trajectory_inference_embedding")
    plt.close(fig)

    # Plot adjacency matrix
    adj_mat = pd.read_csv("../output/ipscs/ipscs_adj_bowside.csv", index_col=0)
    grn_bowside = nx.from_pandas_adjacency(adj_mat, create_using=nx.DiGraph)

    # Remove edges below cutoff
    n_edges = 25  # use top positive and negative edges
    sorted_weights = np.sort(adj_mat.values, axis=None)
    pos_cutoff = sorted_weights[-n_edges]
    neg_cutoff = sorted_weights[n_edges]
    edges_to_remove = [
        (u, v)
        for u, v, d in grn_bowside.edges(data=True)
        if d["weight"] < pos_cutoff and d["weight"] > neg_cutoff
    ]
    grn_bowside.remove_edges_from(edges_to_remove)
    nodes_to_remove = list(nx.isolates(grn_bowside))
    grn_bowside.remove_nodes_from(nodes_to_remove)
    pos = nx.arf_layout(grn_bowside, seed=42)

    # Setup colormap
    vmax = max(abs(adj_mat.values.min()), adj_mat.values.max())
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = sns.cm.vlag

    # Style edges
    edge_colors = {}
    edge_arrowmarkers = []

    for u, v, d in grn_bowside.edges(data=True):
        w = d["weight"]

        # Map weight to seaborn colormap
        edge_colors[(u, v)] = cmap(norm(w))

        # Different arrows for activation and inhibition
        if w > 0:
            edge_arrowmarkers.append(")>")  # standard arrow
        else:
            edge_arrowmarkers.append("|")  # blunt arrow

    # Plot the graph
    fig, ax = plt.subplots(
        figsize=(6.75 * INCH_PER_CM, 8 * INCH_PER_CM), layout="constrained"
    )
    ipx.network(
        grn_bowside,
        pos,  # type: ignore
        node_labels=True,
        style={
            "node": {
                "size": 20,
                "zorder": 2,
                "facecolor": "lightgrey",
                "label_color": "k",
                "label_fontsize": 5,
                "label_fontfamily": "Arial",
            },
            "edge": {
                "zorder": 1,
                "shrink": 1,
                "linewidth": 0.75,
                "arc": True,
                "tension": 0.2,
                "looptension": 3,
                "color": edge_colors,
                "arrow_marker": edge_arrowmarkers,
                "arrow_width": 3,
                "arrow_height": 3,
            },
        },
        ax=ax,
    )
    cax = ax.inset_axes((0.9, 0.05, 0.03, 0.2), transform=ax.transAxes)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("weight", labelpad=1.0)

    sns.despine(fig, left=True, bottom=True)
    fig.savefig(fig_dir / "bowside_grn_graph")
    plt.close()

    # Plot adjacency matrix
    fig, ax = plt.subplots(
        figsize=(10.75 * INCH_PER_CM, 8 * INCH_PER_CM), layout="constrained"
    )
    annot = np.vectorize(lambda x: f"{x:.1f}")(adj_mat.values)
    annot[(annot == "0.0") | (annot == "-0.0")] = ""
    hm = sns.heatmap(
        adj_mat,
        cmap="vlag",
        center=0,
        cbar_kws={
            "location": "bottom",
            "shrink": 0.2,
            "aspect": 10,
            "ticks": [-9, -6, -3, 0, 3],
        },
        annot=annot,
        annot_kws={"fontsize": 5},
        fmt="s",
        ax=ax,
    )
    hm.collections[0].colorbar.set_label("weight", labelpad=1.5)  # type: ignore
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine(fig, top=False, right=False)
    fig.savefig(fig_dir / "bowside_adj_mat")
    plt.close()


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_overview_subfigures()
    make_hematopoiesis_subfigures()
    make_ipsc_subfigures()
