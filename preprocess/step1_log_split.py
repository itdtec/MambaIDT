#!/usr/bin/env python3
"""
step1_log_split.py

Split CERT R4.2 logs by user and scenario using chunked CSV processing.
"""
import os
import argparse
import logging
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    """
    Load processing configuration from a YAML file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def read_chunks(file_path: str, chunk_size: int):
    """
    Generator to read a CSV file in chunks.
    """
    return pd.read_csv(file_path, chunksize=chunk_size)


def save_user_data(chunks, user_id: int, user_dir: str, file_key: str) -> int:
    """
    Filter and append data for a specific user into a CSV under user directory.
    Returns the number of records saved.
    """
    os.makedirs(user_dir, exist_ok=True)
    output_path = os.path.join(user_dir, f"{file_key}.csv")
    total_saved = 0

    for chunk in chunks:
        user_data = chunk[chunk['user'] == user_id]
        if not user_data.empty:
            user_data.to_csv(
                output_path,
                mode='a',
                index=False,
                header=not os.path.exists(output_path)
            )
            total_saved += len(user_data)

    if total_saved:
        logging.info(f"Saved {total_saved} records for user {user_id} in directory {user_dir}")
    else:
        logging.info(f"No records for user {user_id} in directory {user_dir}")

    return total_saved


def process_logs(config: dict):
    """
    Main processing pipeline:
    - Load answers file to map scenarios to users
    - Iterate each log type, scenario, and user, splitting logs accordingly
    """
    answers = pd.read_csv(config['answer_path'])
    filtered = answers[answers['dataset'] == config['dataset']]
    scenario_map = filtered.groupby('scenario')['user'].apply(list).to_dict()

    os.makedirs(config['output_base'], exist_ok=True)

    for key, path in config['file_paths'].items():
        logging.info(f"Processing {key}.csv from {path}")
        for scenario, users in scenario_map.items():
            scenario_dir = os.path.join(config['output_base'], f"scenario_{scenario}")
            for user_id in users:
                user_dir = os.path.join(scenario_dir, str(user_id))
                chunks = read_chunks(path, config.get('chunk_size', 100000))
                save_user_data(chunks, user_id, user_dir, key)


def main():
    parser = argparse.ArgumentParser(description="Split CERT R4.2 logs by user and scenario.")
    parser.add_argument('-c', '--config', required=True, help='YAML config file path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    full_conf = load_config(args.config)
    # Support multi-section or single config
    config = full_conf.get('step1_log_split', full_conf)

    process_logs(config)


if __name__ == '__main__':
    main()
