#!/usr/bin/env python3
"""
stat_feature_engineering.py

Compute daily statistical features from labeled behavior logs.
"""
import os
import argparse
import logging

import pandas as pd
import yaml

from utils_for_feature import (
    extract_domain,
    determine_working_hours,
    add_work_status,
    process_pc_user_data,
    determine_time,
    compute_daily_behavior_counts,
    compute_daily_behavior_duration,
    compute_daily_pc_counts,
    compute_daily_pc_duration,
    compute_daily_time_counts,
    compute_daily_time_duration,
    compute_daily_label,
)

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate daily statistical features.")
    parser.add_argument('-c', '--config', required=True, help='Path to config YAML')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    cfg = load_config(args.config).get('stat_feature_engineering', {})

    input_base = cfg['input_base']
    output_dir = cfg['output_dir']
    scenarios = cfg.get('scenarios', [])
    ldap_df = pd.read_csv(cfg['ldap_path'])
    http_df = pd.read_csv(cfg['http_csv'])

    for scenario in scenarios:
        scenario_dir = os.path.join(input_base, scenario)
        if not os.path.isdir(scenario_dir):
            continue

        for user in os.listdir(scenario_dir):
            file_path = os.path.join(scenario_dir, user, 'behavior_with_label.csv')
            if not os.path.isfile(file_path):
                continue

            try:
                df = pd.read_csv(file_path)

                # 1) Estimate working hours and add status
                start_hr, end_hr = determine_working_hours(df)
                df = add_work_status(df, start_hr, end_hr)

                # 2) Determine PC user type
                df = process_pc_user_data(df, ldap_df)

                # 3) Extract and merge domain categories
                df['url_domain'] = df['url'].apply(lambda u: extract_domain(u) if pd.notna(u) else None)
                df = df.merge(http_df, left_on='url_domain', right_on='domain', how='left')

                # 4) Compute time_diff and date_only
                df = determine_time(df)

                # 5) Compute duration (minutes)
                df['next_time'] = df.groupby('user')['date'].shift(-1)
                df['duration'] = (df['next_time'] - df['date']).dt.total_seconds().div(60).fillna(0)

                # 6) Compute daily statistics
                bc = compute_daily_behavior_counts(df)
                bd = compute_daily_behavior_duration(df)
                pc_counts = compute_daily_pc_counts(df)
                pc_dur = compute_daily_pc_duration(df)
                tc = compute_daily_time_counts(df)
                tdur = compute_daily_time_duration(df)
                lbl = compute_daily_label(df)

                # 7) Merge all statistics
                daily = bc.merge(bd, on='date_only') \
                          .merge(pc_counts, on='date_only') \
                          .merge(pc_dur, on='date_only') \
                          .merge(tc, on='date_only') \
                          .merge(tdur, on='date_only') \
                          .merge(lbl, on='date_only')

                # 8) Save results
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{scenario}_{user}.csv")
                daily.to_csv(output_path, index=False)
                logging.info(f"Saved stats for {scenario}/{user}")

            except Exception as e:
                logging.error(f"Failed processing {scenario}/{user}: {e}")

if __name__ == '__main__':
    main()
