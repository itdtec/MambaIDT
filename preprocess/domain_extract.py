#!/usr/bin/env python3
"""
domain_extract.py

Extract unique domains from HTTP logs using chunked CSV processing.
"""
import os
import argparse
import logging
import pandas as pd
import yaml
from urllib.parse import urlparse


def load_config(path: str) -> dict:
    """
    Load configuration from a YAML file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def extract_domain(url: str) -> str:
    """
    Parse and return the domain from a URL, or None if invalid.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().strip()
        return domain or None
    except Exception:
        return None


def process_http(config: dict):
    """
    Read HTTP log in chunks, extract unique domains, and save to CSV.
    """
    src = config['source_path']
    out = config['output_path']
    chunk_size = config.get('chunk_size', 100000)

    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out), exist_ok=True)

    unique_domains = set()
    for i, chunk in enumerate(pd.read_csv(src, chunksize=chunk_size)):
        if 'url' not in chunk.columns:
            logging.warning(f"Chunk {i+1} missing 'url' column, skipping")
            continue

        chunk['domain'] = chunk['url'].apply(lambda x: extract_domain(x) if pd.notna(x) else None)
        new_domains = chunk['domain'].dropna().unique()
        unique_domains.update(new_domains)
        logging.info(f"Processed chunk {i+1}, total unique domains: {len(unique_domains)}")

    df = pd.DataFrame(sorted(unique_domains), columns=['domain'])
    df.to_csv(out, index=False)
    logging.info(f"Extraction complete, {len(unique_domains)} domains saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Extract unique domains from HTTP logs.")
    parser.add_argument('-c', '--config', required=True, help='YAML config file path')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    full_conf = load_config(args.config)
    # If using a unified config file with multiple sections, extract 'http' section
    config = full_conf.get('domain_extract', full_conf)

    process_http(config)


if __name__ == '__main__':
    main()

