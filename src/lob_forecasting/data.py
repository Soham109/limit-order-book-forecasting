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