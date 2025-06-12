#!/usr/bin/env python3
"""
ldap_user_pc_extract.py

Merge LDAP user info with their most frequently accessed PC, based on logon logs.
"""
import os
import argparse
import logging
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    """
    Load configuration from a YAML file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def find_most_frequent_pc(user_pc_counts: pd.DataFrame) -> pd.Series:
    """
    Determine each user's most frequently accessed PC.
    Returns DataFrame with columns ['user', 'pc'].
    """
    top = (user_pc_counts
           .groupby(['user', 'pc'])
           .size()
           .reset_index(name='counts')
           .sort_values(['user', 'counts'], ascending=[True, False]))
    return top.drop_duplicates(subset=['user'])[['user', 'pc']]


def process_ldap(config: dict):
    """
    Read LDAP and logon files, compute most frequent PC per user, and merge.
    """
    ldap_path = config['ldap_path']
    logon_path = config['logon_path']
    output_path = config['output_path']

    # Load data
    ldap_df = pd.read_csv(ldap_path)
    logon_df = pd.read_csv(logon_path)

    # Compute user-PC counts
    user_pc_counts = (logon_df
                      .groupby(['user', 'pc'])
                      .size()
                      .reset_index(name='counts'))

    # Find top PC per user
    user_top_pc = find_most_frequent_pc(user_pc_counts)

    # Merge with LDAP
    merged = ldap_df.merge(user_top_pc, left_on=config.get('ldap_user_col', 'user_id'),
                           right_on='user', how='left')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save result
    merged.to_csv(output_path, index=False)
    logging.info(f"Saved merged LDAP with PC info to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LDAP info with user top PC from logon logs.")
    parser.add_argument('-c', '--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    full_conf = load_config(args.config)
    config = full_conf.get('device_extract', full_conf)

    process_ldap(config)


if __name__ == '__main__':
    main()
