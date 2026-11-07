"""Binary detection metrics, including the biometric-style EER and HTER, plus participant bootstrap CIs."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def eer(y, score):
    """Equal error rate: the point on the ROC curve where false-accept and false-reject rates meet."""
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def binary_metrics(y, score, threshold=0.0):
    """All headline metrics for scores where `score >= threshold` means the positive (deceptive) class.

    HTER is the half total error rate at the given (pre-specified) threshold: (FAR + FRR) / 2.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den else 0.0
    far = fp / (fp + tn) if fp + tn else float("nan")
    frr = fn / (fn + tp) if fn + tp else float("nan")
    both = y.min() != y.max()
    return {
        "n": int(len(y)),
        "accuracy": float((tp + tn) / len(y)),
        "balanced_accuracy": float((sens + spec) / 2),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "mcc": float(mcc),
        "auc": float(roc_auc_score(y, score)) if both else float("nan"),
        "eer": eer(y, score),
        "hter": float((far + frr) / 2),
    }


def bootstrap_ci(y, score, threshold=0.0, n_boot=2000, alpha=0.05, seed=0):
    """Percentile CIs from resampling units (rows) with replacement, stratified by class."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        draws.append(binary_metrics(y[idx], score[idx], threshold))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.nanpercentile([d[k] for d in draws], lo)),
                float(np.nanpercentile([d[k] for d in draws], hi))]
            for k in draws[0] if k != "n"}


def summarise_repeats(per_repeat):
    """Mean and SD of each metric across repeated cross-validation runs."""
    keys = [k for k in per_repeat[0] if k != "n"]
    return {k: {"mean": float(np.nanmean([m[k] for m in per_repeat])),
                "sd": float(np.nanstd([m[k] for m in per_repeat]))} for k in keys}
