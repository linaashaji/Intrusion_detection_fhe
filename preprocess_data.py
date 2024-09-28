import pandas as pd
from tqdm import tqdm
from pathlib import Path


dataset_path = Path("can-train-and-test-v1.5/hcrl-ch/test_01_DoS/DoS-test.csv")
dataframe = pd.read_csv(dataset_path)
dataframe.sort_values(by="timestamp", ascending=True, inplace=True)


window_size = 50
stride = 20
agg_df = pd.DataFrame(columns=["features", "label"])

for idx in tqdm(range(0, len(dataframe) - window_size + 1, stride)):
    features = dataframe["arbitration_id"].iloc[idx : idx + window_size]
    labels = dataframe["attack"].iloc[idx : idx + window_size]

    df_ = pd.DataFrame(
        [[list(features), labels.sum() > 0]], columns=["features", "label"]
    )
    if len(features) == window_size:
        agg_df = pd.concat([agg_df, df_], ignore_index=True)

agg_df.to_csv(Path("preprocessed_data") / dataset_path.name)
