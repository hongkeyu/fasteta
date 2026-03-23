"""
stage4_parallel.py
Cumulative: Stage 3 + multiprocessing across CPU cores.
"""
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from stage3_zone import run_pipeline as stage3_pipeline


def _worker(args):
    """
    Worker function for parallel execution.
    Processes one DataFrame chunk using the Stage 3 pipeline.
    Single tuple argument because ProcessPoolExecutor.map
    passes one argument per worker call.
    No shared mutable state — zone_lookup is read-only.
    """
    chunk_df, zone_lookup = args
    return stage3_pipeline(chunk_df, zone_lookup)


def run_pipeline(df, zone_lookup=None, n_workers=None):
    """
    Parallel feature engineering — Stage 3 across CPU cores.
    Improvement over Stage 3:
    - DataFrame split into n_workers equal chunks
    - Each chunk processed in a separate OS process
    - Results combined with np.vstack after all workers finish

    Why this works:
    - Feature engineering is embarrassingly parallel — rows are independent
    - Row i does not depend on row j in any stage
    """
    if zone_lookup is None:
        raise ValueError("stage4 requires zone_lookup.")

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    chunks = np.array_split(df, n_workers)  # equal-size chunks
    worker_args = [(chunk, zone_lookup) for chunk in chunks]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_worker, worker_args))

    return np.vstack(results)  # preserves row order
