from .utils_ import  pad_sequence
from .config import load_config

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import json

cfg_set = load_config()["experiment"]

class MambaITDDataset(Dataset):
    def __init__(self, sequence_data, stat_data):
        self.sequence_data = sequence_data
        self.stat_data = stat_data
        self.max_len = cfg_set["max_seq_len"]

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        row_seq = self.sequence_data.iloc[idx]
        row_sta = self.stat_data.iloc[idx]

        S_b = torch.tensor(pad_sequence(row_seq["action_id_seq"], self.max_len), dtype=torch.long)
        S_c = torch.tensor(pad_sequence(row_seq["time_diff_seq"], self.max_len), dtype=torch.float32)

        Y = torch.tensor(row_sta["label"], dtype=torch.float32)

        # 修改点：确保是 float 类型
        feature_values = row_sta.drop(["label", "date_only"])
        feature_values = pd.to_numeric(feature_values, errors='coerce').fillna(0).astype(np.float32).values
        X = torch.tensor(feature_values, dtype=torch.float32)

        return S_b, S_c, X, Y


def split_balanced_data(seq_df, stat_df, train_ratio=0.85, anomaly_aug=4):
    stat_df_cleaned = stat_df.drop(columns=[col for col in ['label', 'date_only'] if col in seq_df.columns])
    merged_df = pd.concat([seq_df, stat_df_cleaned], axis=1)
    merged_df["label"] = stat_df["label"].values
    merged_df["date_only"] = stat_df["date_only"].values
    merged_df = shuffle(merged_df, random_state=42).reset_index(drop=True)

    seq_cols = ["action_id_seq", "time_diff_seq"]
    stat_cols = [col for col in merged_df.columns if col not in seq_cols]
    seq_data_all = merged_df[seq_cols]
    stat_data_all = merged_df[stat_cols]

    # Stratified split
    seq_train, seq_val, stat_train, stat_val = train_test_split(
        seq_data_all, stat_data_all, test_size=1 - train_ratio,
        stratify=stat_data_all["label"], random_state=42
    )

    # Only augment anomalies in training set
    if anomaly_aug > 1:
        df_train = pd.concat([seq_train, stat_train], axis=1)
        pos_df = df_train[df_train["label"] == 1]
        neg_df = df_train[df_train["label"] == 0]
        pos_df_aug = pd.concat([pos_df] * anomaly_aug, ignore_index=True)
        df_train_new = shuffle(pd.concat([neg_df, pos_df_aug], axis=0), random_state=42).reset_index(drop=True)

        seq_train = df_train_new[seq_cols]
        stat_train = df_train_new[stat_cols]

    return (
        seq_train.reset_index(drop=True),
        stat_train.reset_index(drop=True),
        seq_val.reset_index(drop=True),
        stat_val.reset_index(drop=True)
    )