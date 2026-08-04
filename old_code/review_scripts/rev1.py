import sys, numpy as np, pandas as pd
sys.path.insert(0, "/sessions/beautiful-upbeat-curie/mnt/EIGER/Python")
from eiger_core import EigerConfig, call_dmrs_from_matrix, estimate_covariance_threshold
from functions.processing import covariance_blocks, covariance_standard_errors, _cov2

rng = np.random.default_rng(0)

def make_locations(n, chrom="chr1", step=2):
    s = np.arange(n) * step
    return pd.DataFrame({"chr": chrom, "start": s, "end": s + 1})

print("="*70)
print("EXP 1: adjacent covariance vs subgroup prevalence (fixed delta = 0.30)")
print("="*70)
delta = 0.30
noise = 0.03
for n in (20, 100, 519):
    row = []
    for k in (1, 2, 3, 5, 10, n//2):
        f = k / n
        g = np.zeros(n); g[:k] = 1
        cov = []
        for _ in range(300):
            x = 0.5 + delta*g + rng.normal(0, noise, n)
            y = 0.5 + delta*g + rng.normal(0, noise, n)
            cov.append(_cov2(x, y))
        row.append((k, f, np.mean(cov)))
    print(f"n={n}")
    for k, f, c in row:
        print(f"   minor group k={k:>4} (f={f:.4f})  mean adjacent cov = {c:.5f}   "
              f"{'PASS' if c >= 0.012 else 'FAIL'} vs default threshold 0.012")
    # analytic: cov = f(1-f)delta^2
    print(f"   analytic f(1-f)delta^2 for k=3,n={n}: {(3/n)*(1-3/n)*delta**2:.5f}")
print()
print("Minimum delta needed to reach cov=0.012 at prevalence f:")
for f in (0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5):
    print(f"   f={f:<5}  delta_min = {np.sqrt(0.012/(f*(1-f))):.3f}")

print()
print("="*70)
print("EXP 2: probabilistic caller vs cohort size, strong planted DMR")
print("="*70)
def planted(n_samples, n_cpg=400, dmr=slice(100, 130), k=None, delta=0.4, noise=0.03, seed=1):
    r = np.random.default_rng(seed)
    k = k if k is not None else n_samples // 2
    g = np.zeros(n_samples); g[:k] = 1
    M = 0.5 + r.normal(0, noise, (n_cpg, n_samples))
    M[dmr] += delta * g
    return np.clip(M, 0.001, 0.999)

loc = make_locations(400)
for n in (4, 6, 8, 10, 20, 40, 100, 300):
    M = planted(n)
    for method in ("legacy", "probabilistic"):
        cfg = EigerConfig(calling_method=method, min_cpgs=4, covariance_threshold=0.012)
        core, _, _ = call_dmrs_from_matrix(M, loc, config=cfg)
        n_called = 0 if core.empty else (core["call_status"] == "sample_structure").sum()
        print(f"  n={n:>3} {method:<14} blocks={len(core):>3} sample_structure={n_called:>3}", end="")
        if not core.empty:
            print(f"  region_prob={core['region_probability'].max():.3f}"
                  f"  call_prob={np.nanmax(core['call_probability'].values.astype(float)) if 'call_probability' in core else np.nan:.3f}")
        else:
            print()
    print()

print("="*70)
print("EXP 3: |delta|/stdev is nearly delta-free; it measures group imbalance")
print("="*70)
from functions.scoring import descriptive_score
n = 100
for f in (0.03, 0.1, 0.3, 0.5):
    k = max(1, int(f*n))
    for delta in (0.2, 0.4, 0.8):
        g = np.zeros(n); g[:k] = 1
        v = 0.5 + delta*g + rng.normal(0, 0.02, n)
        sd = np.std(v)
        d = np.mean(v[:k]) - np.mean(v[k:])
        print(f"  f={f:<5} delta={delta:<4} |delta|/stdev = {abs(d)/sd:.3f}   "
              f"1/sqrt(f(1-f)) = {1/np.sqrt(f*(1-f)):.3f}")
