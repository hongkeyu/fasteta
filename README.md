# FastETA: Optimizing a Food Delivery Time Prediction Pipeline

**Course:** DS-GA 1019 Advanced Python for Data Science — Spring 2026

**Dataset:** [Kaggle — Food Delivery Time](https://www.kaggle.com/datasets/willianoliveiragibin/food-delivery-time) (~10,000 records)

## Project Goal

Build a food delivery ETA prediction pipeline and optimize it **cumulatively** — each stage keeps all previous optimizations and adds one more. The primary goal is **pipeline speed**, not prediction accuracy. MAE is tracked at every stage to confirm optimizations don't degrade model quality.

## Pipeline Stages

| Stage | Optimization | Key Concept |
|-------|-------------|-------------|
| **Stage 1** | Pure Python baseline | `df.iloc[i]` row-by-row, `math` module haversine |
| **Stage 2** | + NumPy + Numba JIT | Vectorized arrays, `@jit(nopython=True)` compiled haversine |
| **Stage 3** | + Zone Lookup | Precomputed `dict` for O(1) geographic feature retrieval |
| **Stage 4** | + Parallel Processing | `ProcessPoolExecutor` across CPU cores |

## Setup

```bash
# Install dependencies
pip install pandas numpy numba xgboost scikit-learn line_profiler

# Download dataset from Kaggle
# Place Food_Time_new.csv in data/
```

## Usage

```bash
# Step 1: Clean data (run once)
python step0_cleaning.py

# Step 2: Run all benchmarks
python benchmark_runner.py

# Results are written to benchmark_results.txt
```

**Do not run stage files individually.** They only define `run_pipeline()` functions — all timing, profiling, and MAE measurement lives in `benchmark_runner.py`.

## File Structure

```
fasteta/
├── data/
│   └── Food_Time_new.csv          ← raw data (not tracked)
├── step0_cleaning.py               ← run once, produces clean_data.csv
├── stage1_baseline.py              ← pure Python pipeline
├── stage2_numpy_numba.py           ← cumulative: +NumPy +Numba
├── stage3_zone.py                  ← cumulative: +Zone Lookup
├── stage4_parallel.py              ← cumulative: +Parallel
├── model.py                        ← XGBoost train/evaluate (shared)
├── benchmark_runner.py             ← runs everything, writes results
└── benchmark_results.txt           ← auto-generated
```

## Benchmark Results (Jetson Orin Nano, 6 cores)

| Stage | Time (s) | Speedup | MAE (min) |
|-------|----------|---------|-----------|
| Stage 1: Baseline | 0.768 | 1.00x | 15.04 |
| Stage 2: + NumPy + Numba | 0.001 | 739x | 15.04 |
| Stage 3: + Zone Lookup | 0.088 | 8.7x | 14.86 |
| Stage 4: + Parallel | 0.169 | 4.5x | 14.86 |

**Notes:**
- Stage 2 achieves massive speedup via Numba JIT compilation of the haversine loop
- Stage 3 is slower than Stage 2 because zone lookup adds a Python-level loop, but MAE improves (zone features carry predictive signal)
- Stage 4 is slower than Stage 3 at this dataset size (~7k rows) due to process spawn overhead; speedup becomes positive at n ≥ 2000

## Advanced Python Concepts

| Concept | Stage | Course Week |
|---------|-------|-------------|
| cProfile + line_profiler | 1 | Wk 2–4 |
| Function call overhead reduction | 1→2 | Wk 2 |
| NumPy vectorization | 2 | Wk 4 |
| Numba `@jit(nopython=True)` | 2 | Wk 6 |
| Dict O(1) lookup | 3 | Wk 2 |
| `ProcessPoolExecutor` | 4 | Wk 9–10 |
