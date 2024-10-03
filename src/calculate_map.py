import numpy as np
import pandas as pd

from copairs import map, compute


def p_values(dframe: pd.DataFrame, null_size: int, seed: int):
    '''Compute p-values'''
    mask = dframe['n_pos_pairs'] > 0
    pvals = np.full(len(dframe), np.nan, dtype=np.float32)
    scores = dframe.loc[mask, 'average_precision'].values
    null_confs = dframe.loc[mask, ['n_pos_pairs', 'n_total_pairs']].values
    pvals[mask] = compute.p_values(scores, null_confs, null_size, seed)
    return pvals

training_sc_data = pd.read_parquet("../data/processed/training_sc_fs.parquet")
neg_control_sc_data = pd.read_parquet("../data/processed/neg_control_sc_fs_subset.parquet")

training_sc_data["Metadata_control_index"] = -1
neg_control_sc_data["Metadata_control_index"] = neg_control_sc_data.index

assert all([c for c in neg_control_sc_data.columns if c in training_sc_data.columns] == neg_control_sc_data.columns)

training_sc_data = training_sc_data[neg_control_sc_data.columns]
df = pd.concat([training_sc_data, neg_control_sc_data], axis=0)
df = df.query("Metadata_Gene != 'failed QC' and Mitocheck_Phenotypic_Class != 'OutOfFocus'")

assert all(df.filter(regex="^(?!CP__|DP__)").columns == df.filter(regex="^(?!CP__|DP__)").columns)

meta_features = df.filter(regex="^(?!CP__|DP__)").columns
cp_features = df.filter(regex="^(CP__)").columns
dp_features = df.filter(regex="^(DP__)").columns
cp_dp_features = cp_features.tolist() + dp_features.tolist()

data_config = {
    "cp": cp_features,
    "dp": dp_features,
    "cp_dp": cp_dp_features,
}

results = {
    "Metadata_Gene": [],
    "Mitocheck_Phenotypic_Class": [],
}

for grouping in ["Metadata_Gene", "Mitocheck_Phenotypic_Class"]:
    print(f"\nCalculating AP for {grouping}")
    for feature_type in ["cp", "dp"]:
        print(f"\nCalculating AP for {feature_type}")
        ap_result = map.average_precision(
            df[meta_features],
            df[data_config[feature_type][:20]].values,
            pos_sameby=[grouping, "Metadata_control_index"],
            pos_diffby=[],
            neg_sameby=[],
            neg_diffby=[grouping, "Metadata_control_index"],
        ).query("Mitocheck_Phenotypic_Class != 'neg_control'").reset_index(drop=True)

        ap_result["p_value"] = p_values(ap_result, null_size=10000, seed=0)
        ap_result["p < 0.05"] = ap_result["p_value"] < 0.05
        ap_result["-log10(AP p-value)"] = - np.log10(ap_result["p_value"])
        ap_result.rename(columns={"AP": "average_precision"}, inplace=True)
        ap_result["Features"] = feature_type

        results[grouping].append(ap_result)

results = {k: pd.concat(v, axis=0) for k, v in results.items()}
for k, v in results.items():
    v.to_csv(f"../data/processed/{k}_ap_results.csv", index=False)