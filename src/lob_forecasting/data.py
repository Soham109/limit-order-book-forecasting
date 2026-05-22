"""Utilities for loading FI-2010 benchmark data."""

from pathlib import Path
import numpy as np


def load_fi2010_file(path):
    """
    Load one FI-2010 text file and return features and labels.

    FI-2010 files are stored as variables x samples.
    This function transposes the matrix so rows are samples.

    Parameters
    ----------
    path : str or Path
        Path to an FI-2010 .txt file.

    Returns
    -------
    X : np.ndarray
        Feature matrix with shape (n_samples, 144).
    y : np.ndarray
        Label matrix with shape (n_samples, 5).
    """
    path = Path(path)
    raw = np.loadtxt(path)

    if raw.shape[0] != 149:
        raise ValueError(f"Expected 149 rows in raw FI-2010 file, got {raw.shape}")

    data = raw.T
    X = data[:, :144]
    y = data[:, 144:]

    return X, y


def get_fi2010_split_paths(
    *,
    project_root=None,
    data_root=None,
    market="NoAuction",
    normalization="Zscore",
    cf=1,
):
    """
    Build train/test paths for an FI-2010 split.

    Parameters
    ----------
    project_root : str or Path, optional
        Project root used to resolve the default data root.
    data_root : str or Path, optional
        Explicit root for FI-2010 BenchmarkDatasets.
    market : str
        "Auction" or "NoAuction".
    normalization : str
        "Zscore", "MinMax", or "DecPre" (case-insensitive).
    cf : int
        Cross-validation fold index.

    Returns
    -------
    train_file : Path
        Path to the training split file.
    test_file : Path
        Path to the testing split file.
    """
    if data_root is None:
        if project_root is None:
            raise ValueError("Provide project_root or data_root.")
        data_root = Path(project_root) / "data/raw/fi2010/BenchmarkDatasets"

    norm_key = str(normalization).lower()
    norm_map = {
        "zscore": "Zscore",
        "minmax": "MinMax",
        "decpre": "DecPre",
    }
    if norm_key not in norm_map:
        allowed = ", ".join(sorted(norm_map.values()))
        raise ValueError(f"Unknown normalization '{normalization}'. Use one of {allowed}.")

    norm_name = norm_map[norm_key]
    norm_index = {"Zscore": 1, "MinMax": 2, "DecPre": 3}[norm_name]
    file_label = {"Zscore": "ZScore", "MinMax": "MinMax", "DecPre": "DecPre"}[norm_name]

    variant_dir = f"{norm_index}.{market}_{norm_name}"
    split_prefix = f"{market}_{norm_name}"
    train_dir = f"{split_prefix}_Training"
    test_dir = f"{split_prefix}_Testing"

    train_file = (
        Path(data_root)
        / market
        / variant_dir
        / train_dir
        / f"Train_Dst_{market}_{file_label}_CF_{cf}.txt"
    )
    test_file = (
        Path(data_root)
        / market
        / variant_dir
        / test_dir
        / f"Test_Dst_{market}_{file_label}_CF_{cf}.txt"
    )

    return train_file, test_file


def load_fi2010_split(
    *,
    project_root=None,
    data_root=None,
    market="NoAuction",
    normalization="Zscore",
    cf=1,
    verbose=False,
):
    """
    Load an FI-2010 train/test split and optionally report shapes.

    Returns
    -------
    X_train : np.ndarray
        Feature matrix for training.
    y_train : np.ndarray
        Label matrix for training.
    X_test : np.ndarray
        Feature matrix for testing.
    y_test : np.ndarray
        Label matrix for testing.
    """
    train_file, test_file = get_fi2010_split_paths(
        project_root=project_root,
        data_root=data_root,
        market=market,
        normalization=normalization,
        cf=cf,
    )

    X_train, y_train_all = load_fi2010_file(train_file)
    X_test, y_test_all = load_fi2010_file(test_file)

    if verbose:
        print("X_train:", X_train.shape)
        print("y_train_all:", y_train_all.shape)
        print("X_test:", X_test.shape)
        print("y_test_all:", y_test_all.shape)

    return X_train, y_train_all, X_test, y_test_all