import logging
import ast
from typing import Tuple, Dict, Iterator
from urllib.parse import urlparse
import os

import pandas as pd
from scipy.stats import gaussian_kde


def extract_domain(url: str) -> str:
    # Parse and return the domain from a URL, or None if invalid.
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower().strip() or None
    except Exception as e:
        logging.warning(f"Failed to parse URL '{url}': {e}")
        return None


def determine_working_hours(
    df: pd.DataFrame,
    date_col: str = 'date',
    behavior_col: str = 'behavior',
    activity_col: str = 'activity'
) -> Tuple[int, int]:
    # Estimate start/end hours based on Logon/Logoff timestamps.
    df[date_col] = pd.to_datetime(df[date_col], errors='raise')
    mask = (df[behavior_col] == 'logon') & (df[activity_col].isin(['Logon', 'Logoff']))
    logon_df = df.loc[mask]
    if logon_df.empty:
        raise ValueError("No Logon/Logoff records found.")
    on = logon_df.loc[logon_df[activity_col] == 'Logon', date_col]
    off = logon_df.loc[logon_df[activity_col] == 'Logoff', date_col]
    logon_minutes = on.dt.hour.mul(60).add(on.dt.minute)
    logoff_minutes = off.dt.hour.mul(60).add(off.dt.minute)
    if logon_minutes.empty or logoff_minutes.empty:
        raise ValueError("Incomplete Logon or Logoff records.")
    kde_on = gaussian_kde(logon_minutes)
    kde_off = gaussian_kde(logoff_minutes)
    minutes = range(24 * 60)
    start_min = max(minutes, key=lambda m: kde_on(m))
    end_min = max(minutes, key=lambda m: kde_off(m))
    return start_min // 60, (end_min + 59) // 60


def add_work_status(
    df: pd.DataFrame,
    start_hr: int,
    end_hr: int,
    date_col: str = 'date'
) -> pd.DataFrame:
    # Add 'work_status' column based on working hours.
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['hour'] = df[date_col].dt.hour
    df['work_status'] = df['hour'].apply(
        lambda h: 'working_hours' if start_hr <= h < end_hr else 'non_working_hours'
    )
    df.drop(columns=['hour'], inplace=True)
    return df


def process_pc_user_data(
    behavior_df: pd.DataFrame,
    ldap_df: pd.DataFrame,
    user_col: str = 'user',
    pc_col: str = 'pc',
    team_members_col: str = 'team_members',
    supervisor_col: str = 'supervisor_user_id'
) -> pd.DataFrame:
    # Determine pc_user_type for each row.
    df = behavior_df.copy()
    sup_map = ldap_df.set_index(user_col)[supervisor_col].to_dict()
    team_map: Dict[str, list] = {}
    for _, row in ldap_df.iterrows():
        uid = row[user_col]
        try:
            members = ast.literal_eval(row[team_members_col])
        except Exception:
            members = []
        team_map[uid] = members
    def _det(row):
        user = row[user_col]
        pc = row[pc_col]
        sup = sup_map.get(user)
        members = team_map.get(user, [])
        owners = ldap_df.loc[ldap_df[pc_col] == pc, user_col].tolist()
        if not owners:
            return 'other'
        owner = owners[0]
        if owner == user:
            return 'personal'
        if sup and owner == sup:
            return 'supervisor'
        if owner in members:
            return 'department'
        return 'other'
    df['pc_user_type'] = df.apply(_det, axis=1)
    return df


def determine_action(
    row: pd.Series,
    ldap_df: pd.DataFrame,
    http_cat_df: pd.DataFrame,
    behavior_col: str = 'behavior'
) -> str:
    # Generate an action label based on row context.
    beh = row[behavior_col]
    if 'email' in beh:
        sender = row.get('from', '')
        recipients = str(row.get('to', '')).split(';')  # 若有 CC/BCC，可合并： split(';') + split(';') on cc/bcc
        from_type = 'I' if sender.endswith('@dtaa.com') else 'E'
        rec_list = [r for r in recipients if r]
        to_type = 'I' if rec_list and all(r.endswith('@dtaa.com') for r in rec_list) else 'E'
        return f"email_{from_type}2{to_type}"

    elif 'file' in beh:
        filename = str(row.get('filename', '')).lower()
        ext = filename.split('.')[-1]
        if 'activity' in row and pd.notna(row['activity']):
            activity = str(row['activity']).lower()
            if 'open' in activity:
                return f"open_{ext}"
            elif 'write' in activity:
                return f"write_{ext}"
            else:
                raise ValueError(f"error activity: {row['activity']}")

    if 'logon' in beh or 'device' in beh:
        return str(row.get('activity','')).lower()

    if 'http' in beh:
        dom = extract_domain(row.get('url')) if row.get('url') else None
        if dom:
            cat = http_cat_df.loc[http_cat_df['domain'] == dom, 'Category']
            if not cat.empty:
                return cat.iat[0]
        return None
    return None


def compute_cert42_behavior_id(
    behavior: str,
    device: str,
    time_segment: str,
    behavior_map: Dict[str,int],
    device_map: Dict[str,int],
    time_map: Dict[str,int]
) -> int:
    # Encode (behavior, device, time_segment) into a single ID.
    if behavior not in behavior_map:
        raise ValueError(f"Unknown behavior: {behavior}")
    if device not in device_map:
        raise ValueError(f"Unknown device: {device}")
    if time_segment not in time_map:
        raise ValueError(f"Unknown time segment: {time_segment}")
    B, D, TS = behavior_map[behavior], device_map[device], time_map[time_segment] - 1
    N_D, N_TS = len(device_map), len(time_map)
    return B * (N_D * N_TS) + D * N_TS + TS


def determine_time(
    df: pd.DataFrame,
    date_col: str = 'date',
    user_col: str = 'user'
) -> pd.DataFrame:
    # Compute time_diff and date_only columns.
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.sort_values([user_col, date_col], inplace=True)
    df['date_only'] = df[date_col].dt.date
    df['time_diff'] = df.groupby(user_col)[date_col].diff().dt.total_seconds().div(60).fillna(0).astype(int)
    return df


def safe_encode_action_id(
    row: pd.Series,
    behavior_map: Dict[str,int],
    device_map: Dict[str,int],
    time_map: Dict[str,int]
) -> int:
    # Safe wrapper for encoding action IDs.
    try:
        return compute_cert42_behavior_id(
            behavior=row['action'],
            device=row['pc_user_type'],
            time_segment=row['work_status'],
            behavior_map=behavior_map,
            device_map=device_map,
            time_map=time_map
        )
    except Exception as e:
        logging.warning(f"Encoding failed at row {row.name}: {e}")
        return None


def aggregate_daily_sequences(
    df: pd.DataFrame,
    time_col: str = 'time_diff',
    action_col: str = 'action_id_seq',
    label_col: str = 'label'
) -> pd.DataFrame:
    # Aggregate per-date sequences and max label.
    seq = df.groupby('date_only')[time_col].apply(list).rename('time_diff_seq')
    enc = df.groupby('date_only')[action_col].apply(list).rename('action_id_seq')
    lbl = df.groupby('date_only')[label_col].max().rename('label').apply(lambda x: 1 if x>0 else 0)
    return pd.concat([seq, enc, lbl], axis=1).reset_index()


def iter_user_files(
    base_dir: str,
    scenarios: list,
    filename: str = 'behavior_with_label.csv'
) -> Iterator[tuple]:
    # Yield (scenario, user, filepath) for each file.
    for scen in scenarios:
        scen_dir = os.path.join(base_dir, scen)
        if not os.path.isdir(scen_dir): continue
        for user in os.listdir(scen_dir):
            path = os.path.join(scen_dir, user, filename)
            if os.path.isfile(path):
                yield scen, user, path


def load_mappings(config: dict) -> Tuple[dict, dict, dict]:
    # Extract mapping dicts from config.
    m = config['mappings']
    return m['behavior_map'], m['device_map'], m['time_map']


# do for stat
def compute_daily_behavior_counts(df: pd.DataFrame) -> pd.DataFrame:
    df_counts = df.groupby(['date_only','behavior']).size().unstack(fill_value=0).reset_index()
    for col in ['logon','email','http','file','device']:
        if col not in df_counts.columns:
            df_counts[col] = 0
    return df_counts.rename(columns={
        'logon':'logon_count','email':'email_count',
        'http':'http_count','file':'file_count','device':'device_count'
    })


def compute_daily_behavior_duration(df: pd.DataFrame) -> pd.DataFrame:
    df_dur = df.groupby(['date_only','behavior'])['duration'].sum().unstack(fill_value=0).reset_index()
    for col in ['logon','email','http','file','device']:
        if col not in df_dur.columns:
            df_dur[col] = 0
    return df_dur.rename(columns={
        'logon':'logon_duration_min','email':'email_duration_min',
        'http':'http_duration_min','file':'file_duration_min','device':'device_duration_min'
    })


def compute_daily_pc_counts(df: pd.DataFrame) -> pd.DataFrame:
    pc_counts = df.groupby(['date_only','pc_user_type']).size().unstack(fill_value=0).reset_index()
    for col in ['personal','department','supervisor','other']:
        if col not in pc_counts.columns:
            pc_counts[col] = 0
    return pc_counts.rename(columns={
        'personal':'personal_count','department':'department_count',
        'supervisor':'supervisor_count','other':'other_count'
    })


def compute_daily_pc_duration(df: pd.DataFrame) -> pd.DataFrame:
    pc_dur = df.groupby(['date_only','pc_user_type'])['duration'].sum().unstack(fill_value=0).reset_index()
    for col in ['personal','department','supervisor','other']:
        if col not in pc_dur.columns:
            pc_dur[col] = 0
    return pc_dur.rename(columns={
        'personal':'personal_duration','department':'department_duration',
        'supervisor':'supervisor_duration','other':'other_duration'
    })


def compute_daily_time_counts(df: pd.DataFrame) -> pd.DataFrame:
    tc = df.groupby(['date_only','work_status']).size().unstack(fill_value=0).reset_index()
    for col in ['working_hours','non_working_hours']:
        if col not in tc.columns:
            tc[col] = 0
    return tc.rename(columns={
        'working_hours':'working_count','non_working_hours':'non_working_count'
    })


def compute_daily_time_duration(df: pd.DataFrame) -> pd.DataFrame:
    td = df.groupby(['date_only','work_status'])['duration'].sum().unstack(fill_value=0).reset_index()
    for col in ['working_hours','non_working_hours']:
        if col not in td.columns:
            td[col] = 0
    return td.rename(columns={
        'working_hours':'working_duration','non_working_hours':'non_working_duration'
    })


def compute_daily_label(df: pd.DataFrame) -> pd.DataFrame:
    lbl = df.groupby('date_only')['label'].max().reset_index()
    lbl['label'] = lbl['label'].apply(lambda x: 1 if x > 0 else 0)
    return lbl


