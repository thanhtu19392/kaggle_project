import numpy as np
import pandas as pd


NUM_LIFT_BUCKETS = 10


def lift_curve(y_true, y_score):
    idx = np.argsort(y_score)[::-1]
    y_true, y_score = np.array(y_true)[idx], y_score[idx]

    cuts = np.linspace(0, len(y_score) - 1, 1 + NUM_LIFT_BUCKETS, dtype='int')
    sum_by_bucket = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        sum_by_bucket.append([np.sum(y_true[a:b]), b - a])
    sum_by_bucket = np.array(sum_by_bucket)

    mean_by_bucket = 1.0 * sum_by_bucket[:, 0] / sum_by_bucket[:, 1]
    cumsum_ytrue = sum_by_bucket[:, 0].cumsum()

    lift_by_bucket = 1.0 * mean_by_bucket / y_true.mean()
    cumgain = 1.0 * cumsum_ytrue / y_true.sum()
    return pd.DataFrame([
        dict(decile=(i + 1) * 0.1, gain=gain, lift=lift)
        for i, (lift, gain) in enumerate(zip(lift_by_bucket, cumgain))
    ]).set_index('decile')
