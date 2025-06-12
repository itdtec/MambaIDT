#!/usr/bin/env python3
"""
step4_department_relationship_analysis.py

Based on LDAP data, build mappings for supervisor IDs and hierarchical member lists (functional_unit, department, team).
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


def analyze_department(config: dict):
    """
    Read LDAP file, map supervisor names to user IDs, fill missing fields,
    and build member lists for functional_unit, department, and team.
    """
    ldap_path = config['ldap_path']
    output_path = config['output_path']
    name_col = config.get('name_column', 'employee_name')
    user_col = config.get('user_column', 'user_id')
    supervisor_col = config.get('supervisor_column', 'supervisor')

    df = pd.read_csv(ldap_path)

    # Map supervisor names to user IDs
    name_to_id = df.set_index(name_col)[user_col].to_dict()
    df['supervisor_user_id'] = df[supervisor_col].map(name_to_id)

    # Fill missing values
    for field, default in [('functional_unit', 'NO Department'),
                           ('department', 'NO Department'),
                           ('team', 'NO Team')]:
        df[field] = df[field].fillna(default)

    # Build member lists
    fu_members = df.groupby('functional_unit')[user_col].apply(list).to_dict()
    dept_members = df.groupby(['functional_unit', 'department'])[user_col].apply(list).to_dict()
    team_members = df.groupby(['functional_unit', 'department', 'team'])[user_col].apply(list).to_dict()

    df['functional_unit_members'] = df['functional_unit'].map(fu_members)
    df['department_members'] = list(map(lambda r: dept_members.get((r['functional_unit'], r['department'])), df.to_dict('records')))
    df['team_members'] = list(map(lambda r: team_members.get((r['functional_unit'], r['department'], r['team'])), df.to_dict('records')))

    # Ensure output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Saved department relationship data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LDAP department relationships.")
    parser.add_argument('-c', '--config', required=True, help='YAML config file path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    conf_all = load_config(args.config)
    config = conf_all.get('department_extract', conf_all)

    analyze_department(config)


if __name__ == '__main__':
    main()



