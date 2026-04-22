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
# Clone the repo
git clone https://github.com/hongkeyu/fasteta.git
cd fasteta

# Install dependencies
pip install -r requirements.txt

# Step 0 auto-downloads the dataset from Kaggle.
# If you prefer manual download:
#   https://www.kaggle.com/datasets/willianoliveiragibin/food-delivery-time
#   Place the CSV as data/Food_Time_new.csv
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
├── benchmark_results.txt           ← auto-generated
└── requirements.txt                ← pip dependencies
```

## Benchmark Results (6 cores, 7392 rows)

| Stage | Time (s) | Speedup vs Baseline | MAE (min) |
|-------|----------|---------------------|-----------|
| Stage 1: Baseline | 0.768 | 1.00x | 15.04 |
| Stage 2: + NumPy + Numba | 0.001 | 739x | 15.04 |
| Stage 3: + Zone Lookup | 0.005 | 153x | 14.86 |
| Stage 4: + Parallel | 0.117 | 6.6x | 14.86 |

**Notes:**
- Stage 2 achieves massive speedup via Numba JIT compilation of the haversine loop
- Stage 3 uses vectorized integer key encoding (`np.round` + `lat*100000+lon`) to avoid per-row Python `round()`/`tuple()` overhead — 19x faster than naive dict lookup loop
- Stage 3 MAE improves over Stage 2 because zone features carry real predictive signal
- Stage 4 is slower than Stage 3 at this dataset size (~7k rows) due to process spawn overhead (~0.1s fixed cost); speedup becomes positive at n ≥ 2000

## Advanced Python Concepts

| Concept | Stage | Course Week |
|---------|-------|-------------|
| cProfile + line_profiler | 1 | Wk 2–4 |
| Function call overhead reduction | 1→2 | Wk 2 |
| NumPy vectorization | 2 | Wk 4 |
| Numba `@jit(nopython=True)` | 2 | Wk 6 |
| Dict O(1) lookup | 3 | Wk 2 |
| `ProcessPoolExecutor` | 4 | Wk 9–10 |
