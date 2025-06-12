#!/usr/bin/env python3
"""
domain_categories.py

Classify domains into semantic categories (cloud, hacktivist, job_hunting, neutral) based on keyword matching.
"""
import os
import argparse
import logging
import pandas as pd
import yaml

# Define category keyword lists
CATEGORY_KEYWORDS = {
    'web_cloud_storage': ['dropbox', 'box.com', 'drive.google', 'onedrive', 'mega.nz', 'icloud'],
    'web_hacktivist': ['wikileaks', 'leak', 'cryptome'],
    'web_job_hunting': ['linkedin', 'indeed', 'glassdoor', 'monster', 'careerbuilder'],
}


def load_config(path: str) -> dict:
    """
    Load configuration from a YAML file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def classify_domain(domain: str) -> str:
    """
    Assign a category label based on presence of keywords in the domain.
    """
    domain_lower = str(domain).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(key in domain_lower for key in keywords):
            return category
    return 'web_neutral'


def process_classification(config: dict):
    """
    Load domain list, classify each domain, and save labeled output.
    """
    src = config['input_path']
    out = config['output_path']

    # Create output directory if needed
    os.makedirs(os.path.dirname(out), exist_ok=True)

    df = pd.read_csv(src)
    df['Category'] = df['domain'].apply(classify_domain)
    df.to_csv(out, index=False)
    logging.info(f"Classification complete, output saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Classify domains into semantic categories.")
    parser.add_argument('-c', '--config', required=True, help='YAML config file path')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    full_conf = load_config(args.config)
    # Support single or multi-section configs
    config = full_conf.get('domain_categories', full_conf)

    process_classification(config)


if __name__ == '__main__':
    main()
