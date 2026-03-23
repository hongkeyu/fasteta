"""
benchmark_runner.py
Runs all four pipeline stages, captures profiling output,
and writes all results to benchmark_results.txt.

Usage:
    python benchmark_runner.py
"""
import timeit
import cProfile
import pstats
import io
import time
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from stage1_baseline import run_pipeline as stage1
from stage2_numpy_numba import run_pipeline as stage2
from stage3_zone import run_pipeline as stage3, build_zone_lookup
from stage4_parallel import run_pipeline as stage4
from model import train_and_evaluate, RANDOM_STATE, TEST_SIZE

TIMEIT_RUNS = 10
WARMUP_ROWS = 100  # rows for Numba first-call compilation only
COMPLEXITY_N = [500, 1000, 2000, 5000, 10000]
OUTPUT_FILE = "benchmark_results.txt"


def section(f, title):
    f.write("\n" + "=" * 70 + "\n")
    f.write(title + "\n")
    f.write("=" * 70 + "\n\n")


def timed(fn, *args, n=TIMEIT_RUNS):
    """Average time in seconds over n runs."""
    return timeit.timeit(lambda: fn(*args), number=n) / n


def profile_cprofile(fn, *args):
    """Run cProfile, return formatted string of top 20 by cumulative time."""
    pr = cProfile.Profile()
    pr.enable()
    fn(*args)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats('cumulative').print_stats(20)
    return buf.getvalue()


def profile_line(fn, *args):
    """Run line_profiler on fn, return formatted string."""
    try:
        from line_profiler import LineProfiler
        lp = LineProfiler()
        lp(fn)(*args)
        buf = io.StringIO()
        lp.print_stats(stream=buf)
        return buf.getvalue()
    except ImportError:
        return "line_profiler not installed. Run: pip install line_profiler\n"


def main():
    print("Loading clean_data.csv ...")
    df = pd.read_csv("clean_data.csv")
    targets = df['TARGET'].values
    df_feat = df.drop(columns=['TARGET'])

    # build zone_lookup from training rows only — never full dataset
    train_idx, _ = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    zone_lookup = build_zone_lookup(df.iloc[train_idx].reset_index(drop=True))

    # warm up Numba — compile on small subset, not counted in timing
    print("Warming up Numba ...")
    warm = df_feat.iloc[:WARMUP_ROWS]
    stage2(warm)
    stage3(warm, zone_lookup)
    stage4(warm, zone_lookup)
    print("Warm-up done.\n")

    with open(OUTPUT_FILE, 'w') as f:
        # header
        f.write("FastETA Benchmark Results\n")
        f.write(f"Rows: {len(df)} | "
                f"Timeit runs: {TIMEIT_RUNS} | "
                f"Cores: {multiprocessing.cpu_count()} | "
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Section 1: cProfile
        section(f, "SECTION 1: cProfile — Stage 1 Baseline (top 20 by cumulative time)")
        f.write("Identifies which functions consume the most total time.\n"
                "Paste into report appendix.\n\n")
        f.write(profile_cprofile(stage1, df_feat))

        # Section 2: line_profiler
        section(f, "SECTION 2: line_profiler — Stage 1 run_pipeline (line-by-line)")
        f.write("Shows time per line inside the baseline pipeline function.\n"
                "Use this to identify exact lines that are bottlenecks.\n\n")
        from stage1_baseline import run_pipeline as _s1_fn
        f.write(profile_line(_s1_fn, df_feat))

        # Section 3: Pipeline Timing
        section(f, "SECTION 3: Pipeline Timing — All Stages")
        print("Timing all stages ...")
        t1 = timed(stage1, df_feat)
        t2 = timed(stage2, df_feat)
        t3 = timed(stage3, df_feat, zone_lookup)
        t4 = timed(stage4, df_feat, zone_lookup)

        f.write(f"{'Stage':<45} {'Avg Time (s)':>12} {'Speedup':>10}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Stage 1: Baseline (pure Python)':<45} {t1:>12.4f} {'1.00x':>10}\n")
        f.write(f"{'Stage 2: + NumPy + Numba':<45} {t2:>12.4f} {t1/t2:>9.2f}x\n")
        f.write(f"{'Stage 3: + Zone Lookup':<45} {t3:>12.4f} {t1/t3:>9.2f}x\n")
        f.write(f"{'Stage 4: + Parallel Processing':<45} {t4:>12.4f} {t1/t4:>9.2f}x\n")
        f.write(f"\nNote: Numba compilation excluded (warm-up run before timing).\n"
                f"Note: All times averaged over {TIMEIT_RUNS} runs.\n")

        # Section 4: MAE
        section(f, "SECTION 4: XGBoost MAE — All Stages")
        f.write("Same model (n_estimators=100, random_state=42) at every stage.\n"
                "Same train/test split at every stage.\n"
                "MAE should stay flat or improve — never worsen.\n\n")
        print("Evaluating MAE at each stage ...")
        mae1 = train_and_evaluate(stage1(df_feat), targets)
        mae2 = train_and_evaluate(stage2(df_feat), targets)
        mae3 = train_and_evaluate(stage3(df_feat, zone_lookup), targets)
        mae4 = train_and_evaluate(stage4(df_feat, zone_lookup), targets)

        f.write(f"{'Stage':<45} {'MAE (minutes)':>14}\n")
        f.write("-" * 62 + "\n")
        f.write(f"{'Stage 1: Baseline':<45} {mae1:>14.4f}\n")
        f.write(f"{'Stage 2: + NumPy + Numba':<45} {mae2:>14.4f}\n")
        f.write(f"{'Stage 3: + Zone Lookup':<45} {mae3:>14.4f}\n")
        f.write(f"{'Stage 4: + Parallel':<45} {mae4:>14.4f}\n")
        f.write("\nExpected: Stage2 MAE == Stage1 MAE (same features, faster code)\n"
                "Expected: Stage3 MAE <= Stage1 MAE (zone adds predictive signal)\n"
                "Expected: Stage4 MAE == Stage3 MAE (parallel = same computation)\n")

        # Section 4b: Correctness
        section(f, "SECTION 4b: Correctness Verification")
        f.write("Distances from Stage 2/3/4 must match Stage 1 within 0.001 km.\n\n")
        fs1 = stage1(df_feat)
        fs2 = stage2(df_feat)
        fs3 = stage3(df_feat, zone_lookup)
        fs4 = stage4(df_feat, zone_lookup)
        checks = {
            "Stage2 vs Stage1 distances": np.allclose(fs2[:, 0], fs1[:, 0], atol=0.001),
            "Stage3 vs Stage1 distances": np.allclose(fs3[:, 0], fs1[:, 0], atol=0.001),
            "Stage4 vs Stage3 distances": np.allclose(fs4[:, 0], fs3[:, 0], atol=0.001),
        }
        for name, passed in checks.items():
            status = "PASS" if passed else "FAIL — check haversine implementation"
            f.write(f"  {name}: {status}\n")

        # Section 5: Complexity Table
        section(f, "SECTION 5: Complexity — Varying Dataset Size")
        f.write("Shows how execution time scales with n.\n"
                "If Stage 1 grows faster than Stage 4, the complexity difference is real.\n\n")
        f.write(f"{'n':>7} {'Stage1(s)':>10} {'Stage2(s)':>10} "
                f"{'Stage3(s)':>10} {'Stage4(s)':>10} {'S1/S4':>8}\n")
        f.write("-" * 65 + "\n")

        for n in COMPLEXITY_N:
            if n > len(df_feat):
                continue
            sub = df_feat.iloc[:n]
            _ = stage2(sub.iloc[:min(10, n)])  # ensure numba is warm
            _t1 = timeit.timeit(lambda: stage1(sub), number=5) / 5
            _t2 = timeit.timeit(lambda: stage2(sub), number=5) / 5
            _t3 = timeit.timeit(lambda: stage3(sub, zone_lookup), number=5) / 5
            _t4 = timeit.timeit(lambda: stage4(sub, zone_lookup), number=5) / 5
            f.write(f"{n:>7} {_t1:>10.4f} {_t2:>10.4f} "
                    f"{_t3:>10.4f} {_t4:>10.4f} {_t1/_t4:>7.2f}x\n")
            print(f"  n={n:5d} s1={_t1:.4f}s s4={_t4:.4f}s "
                  f"speedup={_t1/_t4:.1f}x")

        # Section 6: Summary
        section(f, "SECTION 6: Summary Table (copy into report)")
        f.write(f"{'Stage':<40} {'Time(s)':>8} {'Speedup':>9} {'MAE(min)':>10}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Stage 1: Baseline (pure Python)':<40} "
                f"{t1:>8.4f} {'1.00x':>9} {mae1:>10.4f}\n")
        f.write(f"{'Stage 2: + NumPy + Numba':<40} "
                f"{t2:>8.4f} {t1/t2:>8.2f}x {mae2:>10.4f}\n")
        f.write(f"{'Stage 3: + Zone Lookup':<40} "
                f"{t3:>8.4f} {t1/t3:>8.2f}x {mae3:>10.4f}\n")
        f.write(f"{'Stage 4: + Parallel Processing':<40} "
                f"{t4:>8.4f} {t1/t4:>8.2f}x {mae4:>10.4f}\n")

    print(f"\nDone. All results written to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
