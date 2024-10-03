from typing import Tuple
from pathlib import Path

import pandas as pd
from pycytominer import feature_select

from utils import load_config, split_data

# config = load_config("../configs/preprocess.yaml")

def setup_data_paths(config: dict) -> Tuple[Path, Path]:
    """
    Setup and return paths based on the provided configuration.
    
    Parameters
    ----------
    config : dict
        Config object with paths.
    
    Returns
    -------
    Tuple[Path, Path]
        Paths to the training and neg_control CSV files.
    """
    data_dir = Path(config['data_dir']).resolve(strict=True)
    training_singlecell_data = data_dir / config['training_singlecell_data']
    neg_control_data = data_dir / config['neg_control_data']
    
    map_out_dir = data_dir / config['map_out_dir']
    map_out_dir.mkdir(parents=True, exist_ok=True)
    
    return training_singlecell_data, neg_control_data


def load_data(training_data_path: Path, neg_control_data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process training and neg_control data from specified paths.
    
    Parameters
    ----------
    training_data_path : Path
        Path to the training data CSV file.
    neg_control_data_path : Path
        Path to the neg_control data CSV file.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The processed training and neg_control DataFrame objects.
    """
    training_sc_data = pd.read_csv(training_data_path).drop("Unnamed: 0", axis=1)
    neg_control_sc_data = pd.read_csv(neg_control_data_path)
    
    neg_control_sc_data.insert(0, "Mitocheck_Phenotypic_Class", "neg_control")
    neg_control_sc_data.insert(1, "Metadata_is_control", 1)
    
    training_sc_data.insert(1, "Metadata_is_control", 0)
    training_sc_data = training_sc_data.drop("Metadata_Object_Outline", axis=1)
    
    print(f"{training_sc_data.shape=}", f"{neg_control_sc_data.shape=}")
    return training_sc_data, neg_control_sc_data


def apply_feature_selection(df: pd.DataFrame, feature_prefix: str) -> pd.DataFrame:
    """
    Apply feature selection to a DataFrame based on specified feature prefix.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    feature_prefix : str
        Prefix of features to be selected.
    feature_selection_func : function
        Function to apply for feature selection, e.g., pycytominer.feature_select.
    
    Returns
    -------
    pd.DataFrame
        DataFrame after feature selection.
    """
    feat_cols = [col for col in df.columns if col.startswith(feature_prefix)]
    meta_df, feature_df = split_data(df, dataset=feature_prefix.strip("_"))
    feature_df = pd.DataFrame(feature_df, columns=feat_cols)
    subtype_df = pd.concat([meta_df, feature_df], axis=1)
    
    subtype_df = feature_select(subtype_df, features=feat_cols)
    
    return subtype_df


fs_ops = [
    'variance_threshold',
    'correlation_threshold',
    'drop_na_columns',
]


# # extracting only CP features
train_meta, train_features = utils.split_data(training_sc_data, dataset="CP")
cp_data = pd.concat([train_meta, pd.DataFrame(train_features)], axis=1)
cp_data.columns = train_meta.columns.tolist() + cp_cols

# applying pycytominer feature select
# had to specify the feature names since the defaults did not match
pycytm_cp_training_feats_df = feature_select(cp_data, features=cp_cols)
pycytm_cp_training_feats_df = pycytm_cp_training_feats_df[
    [
        cols
        for cols in pycytm_cp_training_feats_df.columns.tolist()
        if cols.startswith("CP__")
    ]
]
del cp_data

# now update loaded dataset with pycytominer selected features to trainin dataset
# remove old CP features and added new pycytominer selected CP_features
training_sc_data = training_sc_data[
    [col for col in training_sc_data.columns.tolist() if not col.startswith("CP__")]
]
training_sc_data = pd.concat([training_sc_data, pycytm_cp_training_feats_df], axis=1)
print(f"{training_sc_data.shape=}")

# applying cytominer feature selection negative control data
cp_cols = [
    colname for colname in neg_control_sc_data.columns if colname.startswith("CP__")
]

# extracting only CP features
neg_control_meta, neg_control_features = utils.split_data(
    neg_control_sc_data, dataset="CP"
)
cp_data = pd.concat([neg_control_meta, pd.DataFrame(neg_control_features)], axis=1)
cp_data.columns = neg_control_meta.columns.tolist() + cp_cols

# applying pycytominer feature select
# had to specify the feature names since the defaults did not match
pycytm_cp_training_feats_df = feature_select(cp_data, features=cp_cols)
pycytm_cp_training_feats_df = pycytm_cp_training_feats_df[
    [
        cols
        for cols in pycytm_cp_training_feats_df.columns.tolist()
        if cols.startswith("CP__")
    ]
]
del cp_data

# now update loaded dataset with pycytominer selected features to trainin dataset
# remove old CP features and added new pycytominer selected features
neg_control_sc_data = neg_control_sc_data[
    [col for col in neg_control_sc_data.columns.tolist() if not col.startswith("CP__")]
]
neg_control_sc_data = pd.concat(
    [neg_control_sc_data, pycytm_cp_training_feats_df], axis=1
)
print(f"{neg_control_sc_data.shape=}")

training_sc_data.to_parquet(DATA_DIR / "processed/training_sc_fs.parquet", index=False)
neg_control_sc_data_subset = neg_control_sc_data.sample(frac=0.005, random_state=0)
neg_control_sc_data_subset.to_parquet(DATA_DIR / "processed/neg_control_sc_fs_subset.parquet", index=False)
print(f"{training_sc_data.shape=}", f"{neg_control_sc_data_subset.shape=}")