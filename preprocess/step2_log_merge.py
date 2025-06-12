#!/usr/bin/env python3
"""
step2_log_merge.py

Merge behavior logs per user/scenario.
"""
import os
import argparse
import logging
import pandas as pd
import yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def merge_user_logs(base_dir, scenario, user, data_files):
    user_dir = os.path.join(base_dir, scenario, user)
    frames = []
    for fname in data_files:
        fp = os.path.join(user_dir, fname)
        if os.path.isfile(fp):
            df = pd.read_csv(fp, on_bad_lines='skip')
            df['behavior'] = fname.replace('.csv','')
            frames.append(df)
        else:
            logging.warning(f"Missing {fp}")
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    if 'date' in merged:
        merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
        merged.sort_values('date', inplace=True)
    else:
        logging.warning(f"No 'date' for {user}/{scenario}")
    return merged

def process_all(config):
    base = config['input_base']
    out  = config['output_base']
    scenarios = config['scenarios']
    data_files = config['data_files']
    for scenario in scenarios:
        scen_path = os.path.join(base, scenario)
        if not os.path.isdir(scen_path):
            logging.info(f"Skip missing {scen_path}")
            continue
        for user in os.listdir(scen_path):
            merged = merge_user_logs(base, scenario, user, data_files)
            if merged is None:
                continue
            dest = os.path.join(out, scenario, user)
            os.makedirs(dest, exist_ok=True)
            merged.to_csv(os.path.join(dest,'behavior.csv'), index=False)
            logging.info(f"Saved merged for {user} in {scenario}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', required=True, help='YAML config')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    cfg = load_config(args.config).get('step2_log_merge', {})
    process_all(cfg)

if __name__=='__main__':
    main()
