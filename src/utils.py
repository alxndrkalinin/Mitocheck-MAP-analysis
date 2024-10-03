"""
Contains utility functions to import into the analysis


`split_data()` was develoepd by @roshankern:
https://github.com/WayScience/mitocheck_data/blob/63f37859d993b8de25fefe1cb8a3aac421c3e08a/utils/load_utils.py#L84
"""
from typing import Optional, Union

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def load_config(config_path: Union[Path, str]) -> dict:
    """
    Load configs from a directory or file.

    Parameters
    ----------
    config_path : Union[Path, str]
        Path to config directory or file.

    Returns
    -------
    dict
        Dictionary of configs.
    """
    config_path = Path(config_path)
    if config_path.is_dir():
        config = [OmegaConf.load(sub_conf) for sub_conf in config_path.glob("*.yaml")]
        config = OmegaConf.merge(*config)
    else:
        config = OmegaConf.load(config_path)

    return OmegaConf.to_object(config)


def split_data(pycytominer_output: pd.DataFrame, dataset: str = "CP_and_DP"):
    """
    split pycytominer output to metadata dataframe and np array of feature values

    Parameters
    ----------
    pycytominer_output : pd.DataFrame
        dataframe with pycytominer output
    dataset : str, optional
        which dataset features to split,
        can be "CP" or "DP" or by default "CP_and_DP"

    Returns
    -------
    pd.Dataframe, np.ndarray
        metadata dataframe, feature values

    Credit:
        @roshankern: https://github.com/roshankern
    """
    all_cols = pycytominer_output.columns.tolist()

    # get DP,CP, or both features from all columns depending on desired dataset
    if dataset == "CP":
        feature_cols = [col for col in all_cols if "CP__" in col]
    elif dataset == "DP":
        feature_cols = [col for col in all_cols if "DP__" in col]
    elif dataset == "CP_and_DP":
        feature_cols = [col for col in all_cols if "P__" in col]

    # metadata columns is all columns except feature columns
    metadata_cols = [col for col in all_cols if "P__" not in col]

    metadata_dataframe = pycytominer_output[metadata_cols]
    feature_data = pycytominer_output[feature_cols].values

    return metadata_dataframe, feature_data


def shuffle_by_labels(
    profile: pd.DataFrame, label_col: str, seed: Optional[int] = 1
) -> pd.DataFrame:
    """Shuffles labels in selected column.

    Parameters
    ----------
    profile : pd.DataFrame
        image-based profile with metadata
    target_col : str
        selected column that contains the labels

    Returns
    -------
    pd.DataFrame
        shuffled labeled dataframe
    """

    # type checking
    if not isinstance(profile, pd.DataFrame):
        raise TypeError(f"`profile` must be a dataframe not {type(profile)}")
    if not isinstance(label_col, str):
        raise TypeError(f"`label_col` must be a string type not {type(label_col)}")

    # select column and shuffle labels
    np.random.seed(seed)
    shuffled_labels = np.random.permutation(profile[label_col])
    profile[label_col] = shuffled_labels

    return profile


def shuffle_feature_space(
    profile: pd.DataFrame, col_idx_split: int, seed=1
) -> pd.DataFrame:
    """Shuffled profile's feature space values

    Parameters
    ----------
    feature_val : pd.DataFrame
        _description_
    col_idx_split : int
        column integer where to split the metadata and extracted features
    seed : Optional[int]
        seed seeds in order to maintain reproducibility.

    Returns
    -------
    pd.DataFrame
        feature space shuffled data
    """

    # type checker
    if not (profile, pd.DataFrame):
        raise TypeError(f"`profile` must be a dataframe not {type(profile)}")

    # select
    try:
        feature_vals = profile[profile.columns[col_idx_split:]].astype(float)
    except Exception:
        raise TypeError("The selected index splitter captures non-numerical data")

    # get metadata
    metadata = profile[profile.columns[:col_idx_split]]

    # shuffle feature
    feature_mat = feature_vals.to_numpy()
    for col in feature_mat.T:
        np.random.shuffle(col)

    # reconstruct shuffled data
    feature_shuffled_data = pd.concat(
        [metadata, pd.DataFrame(data=feature_mat)], axis=1
    )
    feature_shuffled_data.columns = profile.columns.tolist()

    # concat metadata with shuffled feature space
    return feature_shuffled_data
