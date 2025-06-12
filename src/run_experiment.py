#!/usr/bin/env python3
from src.config import load_config
from src.dataset import split_balanced_data, MambaITDDataset
from src.training import train_model,validate
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.utils_ import parse_list_string
from sklearn.preprocessing import MinMaxScaler


if __name__=='__main__':
    cfg_set = load_config()["experiment"]
    cfg_path = load_config()["paths"]
    seq_df  = pd.read_csv(cfg_path["seq_csv"])
    stat_df = pd.read_csv(cfg_path["stat_csv"])
    seq_df["action_id_seq"] = seq_df["action_id_seq"].apply(parse_list_string)
    seq_df["time_diff_seq"] = seq_df["time_diff_seq"].apply(parse_list_string)
    # Min-Max normalization on statistics
    numeric_cols = [col for col in stat_df.columns if col not in ['label', 'date_only']]
    stat_df[numeric_cols] = MinMaxScaler().fit_transform(stat_df[numeric_cols])

    seq_train, stat_train, seq_val, stat_val = split_balanced_data(seq_df, stat_df, cfg_set["train_split"], cfg_set["anomaly_aug"])
    train_dataset = MambaITDDataset(seq_train, stat_train)
    val_dataset = MambaITDDataset(seq_val, stat_val)

    model = train_model(train_dataset, val_dataset)

    val_loader = DataLoader(val_dataset, batch_size=cfg_set["batch_size"], shuffle=False)
    validate(model, val_loader)
