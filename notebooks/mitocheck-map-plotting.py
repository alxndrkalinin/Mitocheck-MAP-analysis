#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
from copairs.map import aggregate

warnings.filterwarnings("ignore")


# In[2]:


# Directories
processed_data_dir = pathlib.Path("../data/processed/")
figures_dir = pathlib.Path("../figures/").resolve(strict=True)
sc_ap_scores_dir = (processed_data_dir / "sc_ap_scores").resolve(strict=True)
agg_sc_ap_scores_dir = (processed_data_dir / "aggregate_mAPs").resolve(strict=True)


# ## Preparing the dataset
#
# Next seed of the block, we load the collected mAP single-cell scores generated from the previous notebook.
# These scores are loaded into a dataframe and separated based on the type of features (CP DP CP_DP).
#
# We further divide the feature space based on the type of shuffling methods applied to it.
# Then, we calculate their aggregated average scores using the `copairs` aggregate function.
#

# In[3]:


all_files = list(sc_ap_scores_dir.glob("*.csv"))

cp_sc_mAPs = []
dp_sc_mAPs = []
cp_dp_sc_mAPs = []
for _file in all_files:
    if _file.name.startswith("cp_dp"):
        cp_dp_sc_mAPs.append(pd.read_csv(_file))
    elif _file.name.startswith("cp_"):
        dp_sc_mAPs.append(pd.read_csv(_file))
    elif _file.name.startswith("dp_"):
        cp_sc_mAPs.append(pd.read_csv(_file))

# single-cell mAP scores
cp_sc_mAPs = pd.concat(cp_sc_mAPs)
dp_sc_mAPs = pd.concat(dp_sc_mAPs)
cp_dp_sc_mAPs = pd.concat(cp_dp_sc_mAPs)


# In[4]:


# Separating data frames: One by feature type (CP, DP, CP_DP)
# Additional split is performed using a shuffling approach:
# - feature_shuffled: feature values within the feature space are shuffled.
# - phenotype_shuffled: phenotypic labels are shuffled.

# grabbing all cp features (regular, feature shuffled and labeled shuffled)
reg_cp_sc_mAPs = cp_sc_mAPs.loc[cp_sc_mAPs["shuffled"] == "non-shuffled"]
reg_cp_sc_mAPs["feature_type"] = "CP"
shuffled_feat_cp_sc_mAPs = cp_sc_mAPs.loc[cp_sc_mAPs["shuffled"] == "features_shuffled"]
shuffled_feat_cp_sc_mAPs["feature_type"] = "CP"
shuffled_pheno_cp_sc_mAPs = cp_sc_mAPs.loc[
    cp_sc_mAPs["shuffled"] == "phenotype_shuffled"
]
shuffled_pheno_cp_sc_mAPs["feature_type"] = "CP"

# grabbing all dp features (regular, feature shuffled and labeled shuffled)
reg_dp_sc_mAPs = dp_sc_mAPs.loc[dp_sc_mAPs["shuffled"] == "non-shuffled"]
reg_dp_sc_mAPs["feature_type"] = "DP"
shuffled_feat_dp_sc_mAPs = dp_sc_mAPs.loc[dp_sc_mAPs["shuffled"] == "features_shuffled"]
shuffled_feat_dp_sc_mAPs["feature_type"] = "DP"
shuffled_pheno_dp_sc_mAPs = dp_sc_mAPs.loc[
    dp_sc_mAPs["shuffled"] == "phenotype_shuffled"
]
shuffled_pheno_dp_sc_mAPs["feature_type"] = "DP"

# Grabbing all CP_DP features (regular, features shuffled and labeled shuffled)
reg_cp_dp_sc_mAPs = cp_dp_sc_mAPs.loc[cp_dp_sc_mAPs["shuffled"] == "non-shuffled"]
reg_cp_dp_sc_mAPs["feature_type"] = "CP_DP"
shuffled_feat_cp_dp_sc_mAPs = cp_dp_sc_mAPs.loc[
    cp_dp_sc_mAPs["shuffled"] == "features_shuffled"
]
shuffled_feat_cp_dp_sc_mAPs["feature_type"] = "CP_DP"
shuffled_pheno_cp_dp_sc_mAPs = cp_dp_sc_mAPs.loc[
    cp_dp_sc_mAPs["shuffled"] == "phenotype_shuffled"
]
shuffled_pheno_cp_dp_sc_mAPs["feature_type"] = "CP_DP"


# In[5]:


# Generating sampling_error df
# This table will be used to merge with the aggregate table to get the sampling error a specific category.
merged_sc_ap_scores_df = pd.concat(
    [
        reg_cp_sc_mAPs,
        reg_dp_sc_mAPs,
        shuffled_feat_cp_sc_mAPs,
        shuffled_pheno_cp_sc_mAPs,
        shuffled_feat_dp_sc_mAPs,
        shuffled_pheno_dp_sc_mAPs,
        reg_cp_dp_sc_mAPs,
        shuffled_feat_cp_dp_sc_mAPs,
        shuffled_pheno_cp_dp_sc_mAPs,
    ]
)

# grouping dataframe based on phenotype levels, feature and feature types
df_group = merged_sc_ap_scores_df.groupby(
    by=["Mitocheck_Phenotypic_Class", "feature_type", "shuffled"]
)

# calculating sampling error
sampling_error_df = []
for name, df in df_group:
    pheno, feature_type, shuffled_type = name

    # caclulating sampling error
    avg_percision = df["average_precision"].values
    sampling_error = np.std(avg_percision) / np.sqrt(len(avg_percision))

    sampling_error_df.append([pheno, feature_type, shuffled_type, sampling_error])
cols = ["Mitocheck_Phenotypic_Class", "feature_type", "shuffled", "sampling_error"]
sampling_error_df = pd.DataFrame(sampling_error_df, columns=cols)

# updating name:
sampling_error_df.loc[
    sampling_error_df["shuffled"] == "phenotype_shuffled"
] = "phenotypes_shuffled"

sampling_error_df.head()


# In[6]:


# aggregate single cells scores with cell UUID
data = tuple(merged_sc_ap_scores_df.groupby(by=["Cell_UUID"]))
columns = merged_sc_ap_scores_df.columns
agg_sc_ap_scores_df = []
for cell_id, df1 in data:
    for feature_type, df2 in df1.groupby(by="feature_type"):
        for shuffle_type, df3 in df2.groupby(by="shuffled"):
            aggregated_ap_score = df3["average_precision"].mean()

            # select a single row since all the metadata is the same
            selected_row = df3.iloc[0]

            # update the average precision score of the single row
            selected_row["average_precision"] = aggregated_ap_score
            agg_sc_ap_scores_df.append(selected_row.values.tolist())

# saving into the results repo
agg_sc_ap_scores_df = pd.DataFrame(data=agg_sc_ap_scores_df, columns=columns)
agg_sc_ap_scores_df.to_csv(
    sc_ap_scores_dir / "merged_sc_agg_ap_scores.csv", index=False
)
agg_sc_ap_scores_df.head()


# In[7]:


# Generating aggregate scores with a threshold p-value of 0.05
mAP_dfs = []
for name, df in tuple(agg_sc_ap_scores_df.groupby(by=["feature_type", "shuffled"])):
    agg_df = aggregate(df, sameby=["Mitocheck_Phenotypic_Class"], threshold=0.05)
    agg_df["shuffled"] = name[1]
    agg_df["feature_type"] = name[0]

    mAP_dfs.append(agg_df)

mAP_dfs = pd.concat(mAP_dfs)
mAP_dfs.to_csv(agg_sc_ap_scores_dir / "sc_mAP_scores.csv", index=False)
mAP_dfs.head()


# ## Forming bar plots
#

# ### Forming bar plots with CP Features
#

# In[8]:


# selecting dataset to plot
agg_reg_cp_sc_mAPs = mAP_dfs.loc[
    (mAP_dfs["shuffled"] == "non-shuffled") & (mAP_dfs["feature_type"] == "CP")
]
agg_shuffled_feat_cp_sc_mAPs = mAP_dfs.loc[
    (mAP_dfs["shuffled"] == "features_shuffled") & (mAP_dfs["feature_type"] == "CP")
]
agg_shuffled_pheno_cp_sc_mAPs = mAP_dfs.loc[
    (mAP_dfs["shuffled"] == "phenotype_shuffled") & (mAP_dfs["feature_type"] == "CP")
]

# phenotypes
df = (
    pd.concat(
        [
            agg_reg_cp_sc_mAPs,
            agg_shuffled_feat_cp_sc_mAPs,
            agg_shuffled_pheno_cp_sc_mAPs,
        ]
    )
    .reset_index()
    .drop("index", axis=1)
)[["Mitocheck_Phenotypic_Class", "mean_average_precision", "shuffled"]]


fig = px.bar(
    df,
    x="Mitocheck_Phenotypic_Class",
    y="mean_average_precision",
    color="shuffled",
    barmode="group",
    title="Mean Average Precision for Each Mitocheck Phenotypic Class Using CP Features",
    labels={
        "mean_average_precision": "Mean Average Precision",
        "Mitocheck_Phenotypic_Class": "MitoCheck Phenotypes",
    },
)

fig.show()
fig.write_image(figures_dir / "sc_mAP_cp_barplot.png", width=1200, height=800, scale=3)


# ### Barplot with DP Features
#

# In[9]:


# selecting data to plot
agg_reg_dp_sc_mAPs = mAP_dfs.loc[
    (mAP_dfs["shuffled"] == "non-shuffled") & (mAP_dfs["feature_type"] == "DP")
]
agg_shuffled_feat_dp_sc_mAPs = mAP_dfs.loc[
    (mAP_dfs["shuffled"] == "features_shuffled") & (mAP_dfs["feature_type"] == "DP")
]
agg_shuffled_pheno_dp_sc_mAPs = mAP_dfs.loc[
    (mAP_dfs["shuffled"] == "phenotype_shuffled") & (mAP_dfs["feature_type"] == "DP")
]

# phenotypes
df = (
    pd.concat(
        [
            agg_reg_dp_sc_mAPs,
            agg_shuffled_feat_dp_sc_mAPs,
            agg_shuffled_pheno_dp_sc_mAPs,
        ]
    )
    .reset_index()
    .drop("index", axis=1)
)[["Mitocheck_Phenotypic_Class", "mean_average_precision", "shuffled"]]

fig = px.bar(
    df,
    x="Mitocheck_Phenotypic_Class",
    y="mean_average_precision",
    color="shuffled",
    barmode="group",
    title="Mean Average Precision for Each Mitocheck Phenotypic Class Using CP Features",
    labels={
        "mean_average_precision": "Mean Average Precision",
        "Mitocheck_Phenotypic_Class": "MitoCheck Phenotypes",
    },
)

fig.show()
fig.write_image(figures_dir / "sc_mAP_dp_barplot.png", width=1200, height=800, scale=3)


# ### Barplot with CP_DP Features
#

# In[10]:


# getting data to plot
agg_reg_cp_dp_sc_mAPs = mAP_dfs.loc[
    (mAP_dfs["shuffled"] == "non-shuffled") & (mAP_dfs["feature_type"] == "CP_DP")
]
agg_shuffled_feat_cp_dp_sc_mAPs = mAP_dfs.loc[
    (mAP_dfs["shuffled"] == "feature_shuffled") & (mAP_dfs["feature_type"] == "CP_DP")
]
agg_shuffled_pheno_cp_dp_sc_mAPs = mAP_dfs.loc[
    (mAP_dfs["shuffled"] == "phenotype_shuffled") & (mAP_dfs["feature_type"] == "CP_DP")
]

# phenotypes
data = (
    pd.concat(
        [
            agg_reg_cp_dp_sc_mAPs,
            agg_shuffled_feat_cp_dp_sc_mAPs,
            agg_shuffled_pheno_cp_dp_sc_mAPs,
        ]
    )
    .reset_index()
    .drop("index", axis=1)
)[["Mitocheck_Phenotypic_Class", "mean_average_precision", "shuffled"]]

fig = px.bar(
    df,
    x="Mitocheck_Phenotypic_Class",
    y="mean_average_precision",
    color="shuffled",
    barmode="group",
    title="Mean Average Precision for Each Mitocheck Phenotypic Class Using CP_DP Features",
    labels={
        "mean_average_precision": "Mean Average Precision",
        "Mitocheck_Phenotypic_Class": "MitoCheck Phenotypes",
    },
)

fig.show()
fig.write_image(
    figures_dir / "sc_mAP_cp_dp_barplot.png", width=1200, height=800, scale=3
)


# ## Generating box plots of single cell ap scores

# In[11]:


all_cp_df = pd.concat(
    [
        reg_cp_sc_mAPs,
        shuffled_feat_cp_sc_mAPs,
        shuffled_pheno_cp_sc_mAPs,
    ]
)

all_dp_df = pd.concat(
    [
        reg_dp_sc_mAPs,
        shuffled_pheno_dp_sc_mAPs,
        shuffled_feat_dp_sc_mAPs,
    ]
)

all_cp_dp_df = pd.concat(
    [
        reg_cp_dp_sc_mAPs,
        shuffled_pheno_cp_dp_sc_mAPs,
        shuffled_feat_cp_dp_sc_mAPs,
    ]
)


# In[12]:


# Assuming all_cp_df, all_dp_df, and all_cp_dp_df are your DataFrames
categories_order = all_cp_df["Mitocheck_Phenotypic_Class"].unique()

# Create individual figures with the same category order
fig1 = px.box(
    all_cp_df,
    x="Mitocheck_Phenotypic_Class",
    y="average_precision",
    color="shuffled",
    title="Single Cell Average Percision with CP",
    category_orders={"Mitocheck_Phenotypic_Class": categories_order},
    labels={
        "average_precision": "Average Precision Scores",
        "Mitocheck_Phenotypic_Class": "MitoCheck Phenotypes",
    },
)
fig2 = px.box(
    all_dp_df,
    x="Mitocheck_Phenotypic_Class",
    y="average_precision",
    color="shuffled",
    title="Single Cell Average Percision with DP",
    category_orders={"Mitocheck_Phenotypic_Class": categories_order},
    labels={
        "average_precision": "Average Precision Scores",
        "Mitocheck_Phenotypic_Class": "MitoCheck Phenotypes",
    },
)
fig3 = px.box(
    all_cp_dp_df,
    x="Mitocheck_Phenotypic_Class",
    y="average_precision",
    color="shuffled",
    title="Single Cell Average Percision with CP_DP",
    category_orders={
        "Mitocheck_Phenotypic_Class": categories_order,
    },
    labels={
        "average_precision": "Average Precision Scores",
        "Mitocheck_Phenotypic_Class": "MitoCheck Phenotypes",
    },
)


# In[13]:


fig1.show()
fig1.write_image(
    figures_dir / "sc_APscores_cp_boxplot.png", width=1200, height=800, scale=3
)


# In[14]:


fig2.show()
fig2.write_image(
    figures_dir / "sc_APscores_dp_boxplot.png", width=1200, height=800, scale=3
)


# In[15]:


fig3.show()
fig3.write_image(
    figures_dir / "sc_APscores_cp_dp_boxplot.png", width=1200, height=800, scale=3
)
