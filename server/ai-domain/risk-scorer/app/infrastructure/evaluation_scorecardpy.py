import scorecardpy as sc
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_auc_ks(model, df_woe, y):
    pred = model.predict_proba(df_woe)[:, 1]
    auc = float(roc_auc_score(y, pred))
    pos = np.sort(pred[y == 1])
    neg = np.sort(pred[y == 0])
    all_t = np.unique(np.concatenate([pos, neg]))
    ks = 0.0
    for t in all_t:
        tpr = float((pos <= t).sum()) / float(len(pos)) if len(pos) > 0 else 0.0
        fpr = float((neg <= t).sum()) / float(len(neg)) if len(neg) > 0 else 0.0
        diff = abs(tpr - fpr)
        if diff > ks:
            ks = diff
    return {"auc": auc, "ks": ks}


def compute_psi(train_woe, online_woe, x_cols):
    d1 = train_woe[x_cols]
    d2 = online_woe[x_cols]
    psi = sc.perf_psi(d1, d2)
    return psi
