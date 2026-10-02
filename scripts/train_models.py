import argparse
import itertools
import logging
from functools import partial
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from bowside.dataset import PseudotimeDataset
from bowside.model import GRNModelSDE
from bowside.training import qqloss, sinkhorn_loss, train_integration
from optuna import load_study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def main():
    parser = argparse.ArgumentParser(
        "Training multiple models with the same hyperparameters."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        help="Path to one or more input AnnData objects. If more than one path is "
        "passed, each model will be trained on a different dataset, cycling through the "
        "given datasets and restarting when the end of the list is reached. This should "
        "only be used for multiple datasets with the same genes in the same order, e.g., "
        "subsamples of one dataset, to judge training stability.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output directory. Will be created if it does not exist.",
    )
    parser.add_argument(
        "-n",
        "---name",
        type=str,
        help="Study name for loading optimal hyperparameters.",
    )
    parser.add_argument(
        "--journal-log", type=str, help="Path to journal log of tuning run."
    )
    parser.add_argument(
        "--layer", type=str, help="Layer in adata to use for expression values."
    )
    parser.add_argument(
        "--mask-var", type=str, help="Boolean column in adata.var to filter genes."
    )
    parser.add_argument(
        "--pseudotime-key",
        type=str,
        default="pseudotime",
        help="Key of the pseudotime in adata.obs.",
    )
    parser.add_argument(
        "--time-points",
        type=float,
        nargs="+",
        default=None,
        help="Explicit time points to use for dataset construction.",
    )
    parser.add_argument(
        "--adj-file",
        type=str,
        help="CSV file containg the adjacency list for initialization.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--adj-source-col",
        type=str,
        default="source",
        help="Column containing the source nodes.",
    )
    parser.add_argument(
        "--adj-target-col",
        type=str,
        default="target",
        help="Column containing the target nodes.",
    )
    parser.add_argument(
        "--adj-filter-col",
        type=str,
        help="Column for filtering the adjacency list.",
        default=None,
    )
    parser.add_argument(
        "--adj-filter-value",
        type=str,
        default=None,
        help="Value to use for filtering the adjacency matrix."
        "If this is not set, but the adjacency matrix is filtered, "
        "it will assume boolean values in the filtered column.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="The step size for the solver. This should be chosen "
        "small enough for good numeric integration, but high enough "
        "w.r.t. the time scale of the dataset to reduce compute time.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        required=False,
        default=100,
        help="Patience for early stopping.",
    )
    parser.add_argument(
        "--num-models", type=int, default=10, help="Number of models to train."
    )
    args = parser.parse_args()

    # Load optuna study and setup general hyperparameters
    study = load_study(
        study_name=args.name,
        storage=JournalStorage(JournalFileBackend(args.journal_log)),
    )
    user_attrs = study.user_attrs
    if "num_cells" in user_attrs:
        num_cells = user_attrs["num_cells"]
    else:
        num_cells = 1000
    if "num_quantiles" in user_attrs:
        num_quantiles = user_attrs["num_quantiles"]
    else:
        num_quantiles = 20

    # Load the data
    adatas = [sc.read_h5ad(fname) for fname in args.input]
    datasets = [
        PseudotimeDataset(
            adata=adata,
            layer=args.layer,
            mask_var=args.mask_var,
            pseudotime_key=args.pseudotime_key,
            num_cells=num_cells,
            num_quantiles=num_quantiles,
            time_points=args.time_points,
        )
        for adata in adatas
    ]

    # Load adjacency list if applicable
    adj = None
    if args.adj_file is not None:
        adj_list = pd.read_csv(args.adj_file)
        if args.adj_filter_col is not None:
            if args.adj_filter_value is None:
                adj_list = adj_list[adj_list[args.adj_filter_col]]  # Assume boolean
            else:
                adj_list = adj_list[
                    adj_list[args.adj_filter_col] == args.adj_filter_value
                ]
        adj_list = adj_list.rename(
            columns={args.adj_source_col: "source", args.adj_target_col: "target"}
        )
        adj_list = adj_list[["source", "target"]]

        # Assuming all datasets have the same genes
        adj_list = adj_list[
            (adj_list.source.isin(datasets[0].gene_names))
            & (adj_list.target.isin(datasets[0].gene_names))
        ]
        grn = nx.from_pandas_edgelist(
            adj_list, source="source", target="target", create_using=nx.DiGraph
        )
        for gene in datasets[0].gene_names:
            if gene not in grn.nodes():
                grn.add_node(gene)
        adj = torch.from_numpy(
            nx.to_numpy_array(grn, nodelist=datasets[0].gene_names)
        ).to(torch.float32)

    # Setup the loss function
    hyperparams = study.best_params
    weight = None
    if user_attrs["weighted_loss"] == "True":
        weight = torch.sqrt(
            torch.linspace(0.0, 10.0**2, datasets[0].time_points.size(0))
        )
    if user_attrs["loss_function"] == "sinkhorn":
        loss_fn = sinkhorn_loss
    elif user_attrs["loss_function"] == "qqloss":
        loss_fn = partial(
            qqloss, metric=torch.nn.functional.mse_loss, dim=1, n_quantiles=100
        )
    regularization_strength = user_attrs["regularization_strength"]
    if regularization_strength is None:
        regularization_strength = hyperparams["regularization_strength"]

    # Train the models
    for i, dataset in zip(range(args.num_models), itertools.cycle(datasets)):
        torch.manual_seed(i)
        model = GRNModelSDE(
            dataset.n_genes(),
            adj=adj,
            adj_init_normal=hyperparams["adj_init_normal"] == "True",
            sigma=hyperparams["sigma"],
            shared_sigma=hyperparams["shared_sigma"] == "True",
            fixed_sigma=hyperparams["fixed_sigma"] == "True",
            constant_noise=hyperparams["constant_noise"] == "True",
        )
        optimizer = getattr(torch.optim, hyperparams["optimizer"])(
            model.parameters(), lr=hyperparams["lr"]
        )
        checkpoint_dir = Path(args.output) / f"model_{i}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        log_file = checkpoint_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger = logging.getLogger(f"model_{i}_fitting")
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.info(f"Logger setup complete for fitting model {i}")

        _, _ = train_integration(
            dataset,
            model,
            optimizer,
            loss_fn,
            loss_weights=weight,
            regularization=user_attrs["regularization"],
            regularization_strength=regularization_strength,
            train_proportion=user_attrs["train_proportion"],
            test_proportion=user_attrs["test_proportion"],
            time_point_split=user_attrs["time_point_split"] == "True",
            num_epochs=hyperparams["num_epochs"],
            patience=args.patience,
            solver_method=hyperparams["solver_method"],
            dt=args.dt,
            atol=1e-9,
            checkpoint="final",
            checkpoint_dir=checkpoint_dir,
            logger=logger,
        )

        # Save adjacency matrix separately for convenience
        np.save(checkpoint_dir / "adj_mat.npy", model.adj.detach().cpu().numpy())


if __name__ == "__main__":
    main()
