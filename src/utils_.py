import json
import numpy as np


def parse_list_string(s: str):
    try:
        return json.loads(s)
    except:
        return json.loads(s.strip().replace(" ", ""))


def pad_sequence(seq, max_len, pad_value=0):
    return seq[:max_len] + [pad_value] * (max_len - len(seq))


def otsu_threshold(scores):
    arr = np.array(scores)
    if arr.min() == arr.max():
        return arr.min()
    hist, bins = np.histogram(arr, bins=100)
    mids = 0.5*(bins[:-1] + bins[1:])
    total, best_var, best_thr = len(arr), 0, mids[0]
    for i in range(1, len(hist)):
        p1 = hist[:i].sum()/total; p2 = 1-p1
        if p1 == 0 or p2 == 0: continue
        m1 = (mids[:i]*hist[:i]).sum()/hist[:i].sum()
        m2 = (mids[i:]*hist[i:]).sum()/hist[i:].sum()
        var = p1*p2*(m1-m2)**2
        if var > best_var:
            best_var, best_thr = var, mids[i]
    return best_thr
