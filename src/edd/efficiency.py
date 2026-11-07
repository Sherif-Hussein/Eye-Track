"""Deployment-cost measurements: latency percentiles, throughput, peak memory and model size."""
import pickle
import platform
import time
import tracemalloc

import numpy as np
from threadpoolctl import threadpool_limits


def latency_ms(fn, n_runs=1000, warmup=50):
    """Single-threaded wall-clock latency of `fn()` in milliseconds."""
    with threadpool_limits(1):
        for _ in range(warmup):
            fn()
        t = np.empty(n_runs)
        for i in range(n_runs):
            t0 = time.perf_counter()
            fn()
            t[i] = time.perf_counter() - t0
    t *= 1e3
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "mean": float(t.mean()), "runs": n_runs}


def throughput_per_s(fn, n_items, seconds=2.0):
    """Items processed per second when `fn()` processes `n_items` per call."""
    with threadpool_limits(1):
        calls, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            fn()
            calls += 1
        return float(calls * n_items / (time.perf_counter() - t0))


def peak_memory_bytes(fn):
    """Peak Python-tracked allocation (includes NumPy buffers) during one call of `fn()`."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def model_size_bytes(model):
    return len(pickle.dumps(model))


def svm_parameter_count(svc, n_features):
    """Stored numbers an RBF-SVM needs at inference: support vectors, dual coefficients, intercept."""
    n_sv = int(svc.support_vectors_.shape[0])
    return n_sv * n_features + n_sv + 1


def hardware_info():
    import os
    return {"platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(), "cpu_count": os.cpu_count()}
