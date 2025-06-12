#!/usr/bin/env python3
"""
step3__log_label.py

Label merged user behavior logs with binary labels based on answer files.
"""
import os
import argparse
import logging
import pandas as pd
import csv
import yaml


def load_config(path: str) -> dict:
    """
    Load configuration from a YAML file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def read_answer_ids(filepath: str) -> list:
    """
    Extract all IDs (second column) from an answer CSV.
    """
    ids = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    ids.append(str(row[1]))
    except Exception as e:
        logging.error(f"Failed to read answer file {filepath}: {e}")
    return ids


def label_behavior(config: dict):
    """
    For each scenario and user, label behavior logs based on answer IDs.
    """
    input_base = config['input_base']
    answer_base = config['answer_base']
    scenarios = config['scenarios']
    answer_subfolders = config['answer_subfolders']
    target_columns = config.get('target_columns', [])

    for idx, scenario in enumerate(scenarios):
        scenario_path = os.path.join(input_base, scenario)
        if not os.path.isdir(scenario_path):
            logging.warning(f"Scenario directory not found: {scenario_path}")
            continue
        # Determine corresponding answer folder
        ans_folder = answer_subfolders[idx] if idx < len(answer_subfolders) else None

        for user in os.listdir(scenario_path):
            user_dir = os.path.join(scenario_path, user)
            input_file = os.path.join(user_dir, 'behavior.csv')
            output_file = os.path.join(user_dir, 'behavior_labeled.csv')

            if not os.path.isfile(input_file):
                logging.warning(f"Input file missing: {input_file}")
                continue
            if ans_folder is None:
                logging.warning(f"No answer folder mapping for scenario {scenario}")
                continue

            answer_file = os.path.join(answer_base, ans_folder, f"{ans_folder}-{user}.csv")
            if not os.path.isfile(answer_file):
                logging.warning(f"Answer file missing: {answer_file}")
                continue

            # Read behavior data
            try:
                df = pd.read_csv(input_file)
            except Exception as e:
                logging.error(f"Failed to read {input_file}: {e}")
                continue

            # Initialize label column
            df['label'] = 0
            df['id'] = df['id'].astype(str)

            # Read answer IDs
            ans_ids = set(read_answer_ids(answer_file))
            logging.info(f"Found {len(ans_ids)} answer IDs in {answer_file}")

            # Label matching IDs
            df.loc[df['id'].isin(ans_ids), 'label'] = 1
            matched = df['label'].sum()
            logging.info(f"Matched {matched} IDs for user {user} in {scenario}")

            # Handle unmatched answer IDs
            unmatched = [aid for aid in ans_ids if aid not in set(df['id'])]
            additional = []
            if unmatched:
                logging.warning(f"{len(unmatched)} unmatched IDs for user {user} in {scenario}")
                # Add rows for unmatched answers
                with open(answer_file, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2 and row[1] in unmatched:
                            record = dict.fromkeys(target_columns, None)
                            record['id'] = row[1]
                            record['label'] = 1
                            # Map fields if present
                            # row: [behavior, id, date, user, pc, ...]
                            # behavior is row[0]
                            record['behavior'] = row[0]
                            # Additional parsing could be added here
                            additional.append(record)
                if additional:
                    df = pd.concat([df, pd.DataFrame(additional)], ignore_index=True)
                    logging.info(f"Added {len(additional)} records for unmatched IDs")

            # Sort by date if present
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.sort_values('date', inplace=True)

            # Save output
            os.makedirs(user_dir, exist_ok=True)
            try:
                df.to_csv(output_file, index=False)
                logging.info(f"Saved labeled behavior to {output_file}")
            except Exception as e:
                logging.error(f"Failed to save {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Label user behavior logs with answer IDs.")
    parser.add_argument('-c', '--config', required=True, help='YAML config file path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    full_conf = load_config(args.config)
    config = full_conf.get('step3__log_label', full_conf)

    label_behavior(config)


if __name__ == '__main__':
    main()
