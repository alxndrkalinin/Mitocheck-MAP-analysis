#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pathlib
import sys
from typing import Optional

import numpy as np
import pandas as pd
from copairs.map import run_pipeline
from pycytominer import feature_select

# imports src
sys.path.append("../")
from src import utils  # noqa

# setting up logger
logging.basicConfig(
    filename="map_analysis_testing.log",
    level=logging.DEBUG,
    format="%(levelname)s:%(asctime)s:%(name)s:%(message)s",
)


# ## Helper functions
# Set of helper functions to help out throughout the notebook

# In[2]:


## Helper function
def shuffle_meta_labels(
    dataset: pd.DataFrame, target_col: str, seed: Optional[int] = 0
) -> pd.DataFrame:
    """shuffles labels or values within a single selected column

    Parameters
    ----------
    dataset : pd.DataFrame
        dataframe containing the dataset

    target_col : str
        Column to select in order to conduct the shuffling

    seed : int
        setting random seed

    Returns
    -------
    pd.DataFrame
        shuffled dataset

    Raises
    ------
    TypeError
        raised if incorrect types are provided
    """
    # setting seed
    np.random.seed(seed)

    # type checking
    if not isinstance(target_col, str):
        raise TypeError("'target_col' must be a string type")
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("'dataset' must be a pandas dataframe")

    # selecting column, shuffle values within column, add to dataframe
    dataset[target_col] = np.random.permutation(dataset[target_col].values)
    return dataset


def shuffle_features(feature_vals: np.array, seed: Optional[int] = 0) -> np.array:
    """suffles all values within feature space

    Parameters
    ----------
    feature_vals : np.array
        shuffled

    seed : Optional[int]
        setting random seed

    Returns
    -------
    np.array
        Returns shuffled values within the feature space

    Raises
    ------
    TypeError
        Raised if a numpy array is not provided
    """
    # setting seed
    np.random.seed(seed)

    # shuffle given array
    if not isinstance(feature_vals, np.ndarray):
        raise TypeError("'feature_vals' must be a numpy array")
    if feature_vals.ndim != 2:
        raise TypeError("'feature_vals' must be a 2x2 matrix")

    # creating a copy for feature vales to prevent overwriting of global variables
    feature_vals = np.copy(feature_vals)

    # shuffling feature space
    n_cols = feature_vals.shape[1]
    for col_idx in range(0, n_cols):
        # selecting column, shuffle, and update:
        feature_vals[:, col_idx] = np.random.permutation(feature_vals[:, col_idx])

    return feature_vals


# ## Setting up Paths and loading data

# In[3]:


# parameters
training_singlecell_data = pathlib.Path("../data/raw/training_data.csv.gz").resolve(
    strict=True
)
neg_control_data = pathlib.Path(
    "../data/raw/normalized_data/negative_control_data.csv.gz"
).resolve(strict=True)

# output directories
map_out_dir = pathlib.Path("../data/processed/mAP_scores/")
map_out_dir.mkdir(parents=True, exist_ok=True)


# In[4]:


# training_sc_data = pd.read_parquet("../data/processed/training_sc_data.parquet")
# neg_control_sc_data = pd.read_parquet("../data/processed/neg_control_sc_data.parquet")
training_sc_data = pd.read_csv(training_singlecell_data).drop("Unnamed: 0", axis=1)
neg_control_sc_data = pd.read_csv(neg_control_data)

# adding the Mitocheck_Phenotypic_Class into the controls  and labels
neg_control_sc_data.insert(0, "Mitocheck_Phenotypic_Class", "neg_control")

# adding control labels into the dataset
training_sc_data.insert(1, "Metadata_is_control", 0)
neg_control_sc_data.insert(1, "Metadata_is_control", 1)

# droping column from trainign data since it does not exist in the controls
training_sc_data = training_sc_data.drop("Metadata_Object_Outline", axis=1)


print("control shape:", neg_control_sc_data.shape)
print("training shape:", training_sc_data.shape)


# ## Applying Pycytominer Selected features data

# In[5]:


# applying cytominer feature selection trianing data
cp_cols = [
    colname for colname in training_sc_data.columns if colname.startswith("CP__")
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


# In[7]:


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


# ### mAP Pipeline Parameters

# In[9]:


pos_sameby = [
    "Mitocheck_Phenotypic_Class",
]
pos_diffby = ["Cell_UUID"]

neg_sameby = []
neg_diffby = ["Mitocheck_Phenotypic_Class"]

null_size = 1000
batch_size = 1000

# number of resampling
n_resamples = 10


# ## Running mAP Pipeline on regular dataset

# In[11]:


# storing all map results based on postiive and negative controls and feature types
logging.info("Running mAP pipeline with regular dataset")
map_results_neg_cp = []
map_results_neg_dp = []
map_results_neg_cp_dp = []

# running process
# for loop selects one single phenotype
# then splits the data into metadata and raw feature values
# two different groups that contains 3 splits caused by the types of features
# applie the copairs pipeline
for phenotype in list(training_sc_data["Mitocheck_Phenotypic_Class"].unique()):
    # select training dataset based on phenotype
    selected_training = training_sc_data.loc[
        training_sc_data["Mitocheck_Phenotypic_Class"] == phenotype
    ]
    n_entries_training = selected_training.shape[0]

    # This will generated 100 values [0..100] as seed values
    # This will occur per phenotype
    for seed in range(0, n_resamples):
        # concatenate to positive and negative control
        # selecting around 0.015% of the control data, that's around 117 cells
        training_w_neg = pd.concat(
            [
                selected_training,
                neg_control_sc_data.sample(frac=0.010, random_state=seed).iloc[
                    :n_entries_training
                ],
            ]
        )

        # spliting metadata and raw feature values
        logging.info("splitting data set into metadata and raw feature values")
        negative_training_cp_meta, negative_training_cp_feats = utils.split_data(
            training_w_neg, dataset="CP"
        )
        negative_training_dp_meta, negative_training_dp_feats = utils.split_data(
            training_w_neg, dataset="DP"
        )
        negative_training_cp_dp_meta, negative_training_cp_dp_feats = utils.split_data(
            training_w_neg, dataset="CP_and_DP"
        )

        # placing under "try" block as some phenotype may raise "DivisionByZeroError"
        try:
            # execute pipeline on negative control with trianing dataset with cp features
            logging.info(f"Running pipeline on CP features using {phenotype} phenotype")
            cp_negative_training_result = run_pipeline(
                meta=negative_training_cp_meta,
                feats=negative_training_cp_feats,
                pos_sameby=pos_sameby,
                pos_diffby=pos_diffby,
                neg_sameby=neg_sameby,
                neg_diffby=neg_diffby,
                batch_size=batch_size,
                null_size=null_size,
            )

            # adding columsn
            cp_negative_training_result["shuffled"] = "non-shuffled"
            cp_negative_training_result["seed_val"] = seed

            # append to list
            map_results_neg_cp.append(cp_negative_training_result)

            # execute pipeline on negative control with trianing dataset with dp features
            logging.info(f"Running pipeline on DP features using {phenotype} phenotype")
            dp_negative_training_result = run_pipeline(
                meta=negative_training_dp_meta,
                feats=negative_training_dp_feats,
                pos_sameby=pos_sameby,
                pos_diffby=pos_diffby,
                neg_sameby=neg_sameby,
                neg_diffby=neg_diffby,
                batch_size=batch_size,
                null_size=null_size,
            )

            # adding shuffle label column
            dp_negative_training_result["shuffled"] = "non-shuffled"
            dp_negative_training_result["seed_val"] = seed

            # append to list
            map_results_neg_dp.append(dp_negative_training_result)

            # execute pipeline on negative control with trianing dataset with cp_dp features
            logging.info(
                f"Running pipeline on CP and DP features using {phenotype} phenotype"
            )
            cp_dp_negative_training_result = run_pipeline(
                meta=negative_training_cp_dp_meta,
                feats=negative_training_cp_dp_feats,
                pos_sameby=pos_sameby,
                pos_diffby=pos_diffby,
                neg_sameby=neg_sameby,
                neg_diffby=neg_diffby,
                batch_size=batch_size,
                null_size=null_size,
            )

            # adding shuffle label column
            cp_dp_negative_training_result["shuffled"] = "non-shuffled"
            cp_dp_negative_training_result["seed_val"] = seed

            # append to list
            map_results_neg_cp_dp.append(cp_dp_negative_training_result)
        except ZeroDivisionError as e:
            logging.warning(f"{e} captured on phenotye: {phenotype}. Skipping")
            continue


# concatenating all datasets
pd.concat(map_results_neg_cp).to_csv(
    map_out_dir / "cp_sc_mAP_scores_regular.csv", index=False
)
pd.concat(map_results_neg_dp).to_csv(
    map_out_dir / "dp_sc_mAP_scores_regular.csv", index=False
)
pd.concat(map_results_neg_cp_dp).to_csv(
    map_out_dir / "cp_dp_sc_mAP_scores_regular.csv", index=False
)


# ## Running MAP Pipeline with shuffled phenotype labels

# In[12]:


logging.info("Running mAP pipeline with shuffled phenotype labeled data")

# storing generated mAP pipline results seperated by feature
shuffled_labels_map_results_neg_cp = []
shuffled_labels_map_results_neg_dp = []
shuffled_labels_map_results_neg_cp_dp = []

# running process
# for loop selects one single phenotype
# then splits the data into metadata and raw feature values
# two different groups that contains 3 splits caused by the types of features
# applie the copairs pipeline
for phenotype in list(training_sc_data["Mitocheck_Phenotypic_Class"].unique()):
    # select training dataset based on phenotype
    logging.info(f"Phenotype selected: {phenotype}")
    selected_training = training_sc_data.loc[
        training_sc_data["Mitocheck_Phenotypic_Class"] == phenotype
    ]
    n_entries_training = selected_training.shape[0]

    # This will generated 100 values [0..100] as seed values
    # seed values will
    for seed in range(0, n_resamples):
        # setting seed
        logging.info(
            f"Running MAP Pipeline with shuffled data. Setting random seed too: {seed}"
        )
        np.random.seed(seed)

        # Below, we are running the same test, but we are shuffling the phenotypes
        logging.info(
            "Shuffling data based on the Mitocheck_Phenotypic_Class (phenotype) labels"
        )

        # concatenate to positive and negative control and shuffle labels
        training_w_neg = pd.concat(
            [
                selected_training,
                neg_control_sc_data.sample(frac=0.010, random_state=seed).iloc[
                    :n_entries_training
                ],
            ]
        )
        training_w_neg = shuffle_meta_labels(
            dataset=training_w_neg, target_col="Mitocheck_Phenotypic_Class", seed=seed
        )

        # splitting metadata labeled shuffled data
        logging.info("splitting shuffled data set into metadata and raw feature values")
        (
            shuffled_negative_training_cp_meta,
            shuffled_negative_training_cp_feats,
        ) = utils.split_data(training_w_neg, dataset="CP")
        (
            shuffled_negative_training_dp_meta,
            shuffled_negative_training_dp_feats,
        ) = utils.split_data(training_w_neg, dataset="DP")
        (
            shuffled_negative_training_cp_dp_meta,
            shuffled_negative_training_cp_dp_feats,
        ) = utils.split_data(training_w_neg, dataset="CP_and_DP")

        try:
            # execute pipeline on negative control with trianing dataset with cp features
            logging.info(
                f"Running pipeline on CP features using {phenotype} phenotype, data is shuffled by phenoptype labels"
            )
            shuffled_cp_negative_training_result = run_pipeline(
                meta=shuffled_negative_training_cp_meta,
                feats=shuffled_negative_training_cp_feats,
                pos_sameby=pos_sameby,
                pos_diffby=pos_diffby,
                neg_sameby=neg_sameby,
                neg_diffby=neg_diffby,
                batch_size=batch_size,
                null_size=null_size,
            )

            # adding shuffle label column
            shuffled_cp_negative_training_result["shuffled"] = "phenotype_shuffled"
            shuffled_cp_negative_training_result["seed_val"] = seed

            # append to list
            shuffled_labels_map_results_neg_cp.append(
                shuffled_cp_negative_training_result
            )

            # execute pipeline on negative control with trianing dataset with dp features
            logging.info(
                f"Running pipeline on DP features using {phenotype} phenotype, data is shuffled by phenoptype labels"
            )
            shuffled_dp_negative_training_result = run_pipeline(
                meta=shuffled_negative_training_dp_meta,
                feats=shuffled_negative_training_dp_feats,
                pos_sameby=pos_sameby,
                pos_diffby=pos_diffby,
                neg_sameby=neg_sameby,
                neg_diffby=neg_diffby,
                batch_size=batch_size,
                null_size=null_size,
            )

            # adding shuffle label column
            shuffled_dp_negative_training_result["shuffled"] = "phenotype_shuffled"
            shuffled_dp_negative_training_result["seed_val"] = seed

            # append to list
            shuffled_labels_map_results_neg_dp.append(
                shuffled_dp_negative_training_result
            )

            # execute pipeline on negative control with trianing dataset with cp_dp features
            logging.info(
                f"Running pipeline on CP and DP features using {phenotype} phenotype, data is shuffled by phenoptype labels"
            )
            shuffled_cp_dp_negative_training_result = run_pipeline(
                meta=shuffled_negative_training_cp_dp_meta,
                feats=shuffled_negative_training_cp_dp_feats,
                pos_sameby=pos_sameby,
                pos_diffby=pos_diffby,
                neg_sameby=neg_sameby,
                neg_diffby=neg_diffby,
                batch_size=batch_size,
                null_size=null_size,
            )

            # adding shuffle label column
            shuffled_cp_dp_negative_training_result["shuffled"] = "phenotype_shuffled"
            shuffled_cp_dp_negative_training_result["seed_val"] = seed

            # append to list
            shuffled_labels_map_results_neg_cp_dp.append(
                shuffled_cp_dp_negative_training_result
            )
        except ZeroDivisionError as e:
            logging.warning(f"{e} captured on phenotye: {phenotype}. Skipping")
            continue

# saving to csv
pd.concat(shuffled_labels_map_results_neg_cp).to_csv(
    map_out_dir / "cp_sc_mAP_scores_label_shuffled.csv", index=False
)
pd.concat(shuffled_labels_map_results_neg_dp).to_csv(
    map_out_dir / "dp_sc_mAP_scores_label_shuffled.csv", index=False
)
pd.concat(shuffled_labels_map_results_neg_cp_dp).to_csv(
    map_out_dir / "cp_dp_sc_mAP_scores_label_shuffled.csv", index=False
)


# ## Running MAP Pipeline with shuffled feature space

# In[13]:


logging.info("Running mAP pipeline with shuffled feature space data")
shuffled_feat_map_results_neg_cp = []
shuffled_feat_map_results_neg_dp = []
shuffled_feat_map_results_neg_cp_dp = []

# running process
# for loop selects one single phenotype
# then splits the data into metadata and raw feature values
# two different groups that contains 3 splits caused by the types of features
# applie the copairs pipeline
for phenotype in list(training_sc_data["Mitocheck_Phenotypic_Class"].unique()):
    # select training dataset based on phenotype
    selected_training = training_sc_data.loc[
        training_sc_data["Mitocheck_Phenotypic_Class"] == phenotype
    ]
    n_entries_training = selected_training.shape[0]

    # This will generated 100 values [0..100] as seed values
    # seed values will
    for seed in range(0, n_resamples):
        # setting seed
        logging.info(
            f"Running MAP Pipeline with shuffled data. Setting random seed too: {seed}"
        )
        np.random.seed(seed)

        # Below, we are running the same test, but we are shuffling the phenotypes
        logging.info("Shuffling data based on the feature space")
        training_w_neg = pd.concat(
            [
                selected_training,
                neg_control_sc_data.sample(frac=0.010, random_state=seed).iloc[
                    :n_entries_training
                ],
            ]
        )

        # split the shuffled dataset
        # spliting metadata and raw feature values
        logging.info("splitting shuffled data set into metadata and raw feature values")
        (
            shuffled_negative_training_cp_meta,
            shuffled_negative_training_cp_feats,
        ) = utils.split_data(training_w_neg, dataset="CP")
        (
            shuffled_negative_training_dp_meta,
            shuffled_negative_training_dp_feats,
        ) = utils.split_data(training_w_neg, dataset="DP")
        (
            shuffled_negative_training_cp_dp_meta,
            shuffled_negative_training_cp_dp_feats,
        ) = utils.split_data(training_w_neg, dataset="CP_and_DP")

        # shuffling the features, this will overwrite the generated feature space from above with the shuffled one
        shuffled_negative_training_cp_feats = shuffle_features(
            feature_vals=shuffled_negative_training_cp_feats, seed=seed
        )
        shuffled_negative_training_dp_feats = shuffle_features(
            feature_vals=shuffled_negative_training_dp_feats, seed=seed
        )
        shuffled_negative_training_cp_dp_feats = shuffle_features(
            feature_vals=shuffled_negative_training_cp_dp_feats, seed=seed
        )

        try:
            # execute pipeline on negative control with trianing dataset with cp features
            logging.info(
                f"Running pipeline on CP features using {phenotype} phenotype, feature space is shuffled"
            )
            shuffled_cp_feat_negative_training_result = run_pipeline(
                meta=shuffled_negative_training_cp_meta,
                feats=shuffled_negative_training_cp_feats,
                pos_sameby=pos_sameby,
                pos_diffby=pos_diffby,
                neg_sameby=neg_sameby,
                neg_diffby=neg_diffby,
                batch_size=batch_size,
                null_size=null_size,
            )

            # adding shuffle label column
            shuffled_cp_feat_negative_training_result["shuffled"] = "features_shuffled"
            shuffled_cp_feat_negative_training_result["seed_val"] = seed

            # append to list
            shuffled_feat_map_results_neg_cp.append(
                shuffled_cp_feat_negative_training_result
            )

            # execute pipeline on negative control with trianing dataset with dp features
            logging.info(
                f"Running pipeline on DP features using {phenotype} phenotype, feature space is shuffled"
            )
            shuffled_dp_feat_negative_training_result = run_pipeline(
                meta=shuffled_negative_training_dp_meta,
                feats=shuffled_negative_training_dp_feats,
                pos_sameby=pos_sameby,
                pos_diffby=pos_diffby,
                neg_sameby=neg_sameby,
                neg_diffby=neg_diffby,
                batch_size=batch_size,
                null_size=null_size,
            )

            # adding shuffle label column
            shuffled_dp_feat_negative_training_result["shuffled"] = "features_shuffled"
            shuffled_dp_feat_negative_training_result["seed_val"] = seed

            # append to list
            shuffled_feat_map_results_neg_dp.append(
                shuffled_dp_feat_negative_training_result
            )

            # execute pipeline on negative control with trianing dataset with cp_dp features
            logging.info(
                f"Running pipeline on CP and DP features using {phenotype} phenotype, feature space is shuffled"
            )
            shuffled_cp_dp_feat_negative_training_result = run_pipeline(
                meta=shuffled_negative_training_cp_dp_meta,
                feats=shuffled_negative_training_cp_dp_feats,
                pos_sameby=pos_sameby,
                pos_diffby=pos_diffby,
                neg_sameby=neg_sameby,
                neg_diffby=neg_diffby,
                batch_size=batch_size,
                null_size=null_size,
            )

            # adding shuffle label column
            shuffled_cp_dp_feat_negative_training_result[
                "shuffled"
            ] = "features_shuffled"
            shuffled_cp_dp_feat_negative_training_result["seed_val"] = seed

            # append to list
            shuffled_feat_map_results_neg_cp_dp.append(
                shuffled_cp_dp_feat_negative_training_result
            )
        except ZeroDivisionError as e:
            logging.warning(f"{e} captured on phenotype: {phenotype}. Skipping")
            continue


# saving to csv
pd.concat(shuffled_feat_map_results_neg_cp).to_csv(
    map_out_dir / "cp_sc_mAP_scores_feat_shuffled.csv", index=False
)
pd.concat(shuffled_feat_map_results_neg_dp).to_csv(
    map_out_dir / "dp_sc_mAP_scores_feat_shuffled.csv", index=False
)
pd.concat(shuffled_feat_map_results_neg_cp_dp).to_csv(
    map_out_dir / "cp_dp_sc_mAP_scores_feat_shuffled.csv", index=False
)
