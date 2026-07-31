import argparse
import re

import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Generating transcription factor adjacency list, with edges "
        "from each transcription factor to all genes."
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Path to input CSV file with adjacency list."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="adj_list_full.csv.gz",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--source-col",
        type=str,
        default="source",
        help="Column containing the source nodes, used as transcription factors.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Column containing the target nodes, used to define all genes.",
    )
    args = parser.parse_args()

    # Read original adjacency list
    adj_list = pd.read_csv(args.input)

    # Get transcription factors and all genes
    transcription_factors = set(adj_list[args.source_col].unique())
    all_genes = transcription_factors.union(adj_list[args.target_col].unique())
    # Only include genes starting with alphanumeric character
    transcription_factors = sorted(
        [tf for tf in transcription_factors if re.match(r"^\w", tf)]
    )
    all_genes = sorted([g for g in all_genes if re.match(r"^\w", g)])

    # Construct adjacency matrix
    adj_matrix = pd.DataFrame(
        np.ones((len(transcription_factors), len(all_genes))),
        index=transcription_factors,
        columns=all_genes,
    )
    adj_matrix.index.name = "source"
    adj_list = (
        adj_matrix.reset_index()
        .melt(id_vars=["source"], var_name="target")
        .drop(columns="value")
    )
    adj_list.to_csv(args.output, index=False, header=True)
