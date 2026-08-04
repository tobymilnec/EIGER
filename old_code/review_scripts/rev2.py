import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/sessions/beautiful-upbeat-curie/mnt/EIGER/Python")
from eiger_core import (EigerConfig, call_dmrs_from_matrix, estimate_covariance_threshold,
                        covariance_edge_support, smooth_edge_support, _effect_probability,
                        _probabilistic_separation_from_values, beta_measurement_variance)
from functions.processing import _cov2, covariance_standard_errors, covariance_standard_errors_with_measurement

def loc_df(n, chrom="chr1", step=2):
    s = np.arange(n) * step
    return pd.DataFrame({"chr": chrom, "start": s, "end": s + 1})

print("="*72)
print("EXP 4: NULL data (no DMR anywhere). False discovery behaviour.")
print("="*72)
for n in (10, 30, 100):
    for method in ("legacy", "probabilistic"):
        blocks = calls = 0; cps = []; qs = []
        for rep in range(20):
            r = np.random.default_rng(1000 + rep)
            M = np.clip(0.5 + r.normal(0, 0.10, (2000, n)), .001, .999)   # iid noise, sd 0.10
            cfg = EigerConfig(calling_method=method, min_cpgs=4, covariance_threshold=0.012)
            core, _, _ = call_dmrs_from_matrix(M, loc_df(2000), config=cfg)
            blocks += len(core)
            if not core.empty:
                calls += int((core["call_status"] == "sample_structure").sum())
                if method == "probabilistic":
                    cps += list(pd.to_numeric(core["call_probability"], errors="coerce").dropna())
                qs += list(pd.to_numeric(core["sample_qvalue_descriptive"], errors="coerce").dropna())
        print(f"  n={n:>4} {method:<14} blocks/20reps={blocks:>4}  sample_structure={calls:>4}"
              f"  q<0.05: {np.mean(np.array(qs) < 0.05) if qs else float('nan'):.2f}"
              f"  mean call_prob={np.mean(cps) if cps else float('nan'):.3f}")

print()
print("="*72)
print("EXP 5: shared latent factor (cell composition) -> auto threshold null is wrong")
print("="*72)
n, ncpg = 60, 5000
r = np.random.default_rng(7)
lat = r.normal(0, 1, n)                       # e.g. tumour purity / immune fraction
load = r.normal(0, 0.08, (ncpg, 1))           # CpG-specific loading, NO regional structure
M = np.clip(0.5 + load * lat + r.normal(0, 0.05, (ncpg, n)), .001, .999)
cov = np.array([_cov2(M[i], M[i+1]) for i in range(ncpg-1)])
elig = np.ones(ncpg-1, bool)
est = estimate_covariance_threshold(M, elig, quantile=0.995, permutations=20, max_pairs=20000, seed=1)
print(f"  true adjacent cov (latent-factor only, no DMRs): mean={cov.mean():.5f}  "
      f"q95={np.quantile(cov,.95):.5f}  q99.5={np.quantile(cov,.995):.5f}")
print(f"  EIGER auto threshold from its resampling null      = {est.threshold:.5f}")
print(f"  fraction of NULL genome edges above that threshold = {np.mean(cov >= est.threshold):.3f}")
cfg = EigerConfig(calling_method="probabilistic", covariance_threshold_method="auto", min_cpgs=4)
core, _, _ = call_dmrs_from_matrix(M, loc_df(ncpg), config=cfg)
print(f"  DMR blocks called on this DMR-free matrix          = {len(core)}"
      f"  ({(core['call_status']=='sample_structure').sum() if len(core) else 0} 'sample_structure')")

print()
print("="*72)
print("EXP 6: sample-bootstrap deflates covariance by (n-1)/n -> stability biased low")
print("="*72)
for n in (8, 20, 50):
    r = np.random.default_rng(3)
    x = r.normal(0, .1, n); y = x + r.normal(0, .02, n)
    obs = _cov2(x, y)
    b = [ _cov2(x[s], y[s]) for s in (np.random.default_rng(i).integers(0, n, n) for i in range(20000)) ]
    print(f"  n={n:>3} observed cov={obs:.5f}  bootstrap mean={np.mean(b):.5f}  "
          f"ratio={np.mean(b)/obs:.4f}   (n-1)/n={(n-1)/n:.4f}")

print()
print("="*72)
print("EXP 7: shared reads make adjacent measurement error correlated -> false covariance")
print("="*72)
n, ncpg, depth = 40, 3000, 8
r = np.random.default_rng(11)
truth = np.full((ncpg, n), 0.5)          # NO biological variation at all
# independent-read model
mA = r.binomial(depth, truth) / depth
covA = np.array([_cov2(mA[i], mA[i+1]) for i in range(ncpg-1)])
# shared-read model: same reads cover both CpGs of a pair (realistic at <150bp)
mB = mA.copy()
for i in range(0, ncpg-1, 2):
    mB[i+1] = mB[i]                       # perfectly shared fragment support
covB = np.array([_cov2(mB[i], mB[i+1]) for i in range(0, ncpg-1, 2)])
print(f"  independent reads: mean adjacent cov = {covA.mean():.5f}  "
      f"frac >= 0.012: {np.mean(covA>=0.012):.3f}")
print(f"  shared reads     : mean adjacent cov = {covB.mean():.5f}  "
      f"frac >= 0.012: {np.mean(covB>=0.012):.3f}   <-- pure sequencing noise, zero biology")
print(f"  (binomial variance at depth {depth}, p=0.5 = {0.25/depth:.5f})")

print()
print("="*72)
print("EXP 8a: SE small-sample correction is applied to the SE, not the variance")
print("="*72)
for n in (5, 10, 30, 100):
    r = np.random.default_rng(5)
    true_cov = 0.02
    est = []
    for _ in range(4000):
        z = r.normal(0, 1, n)
        x = np.sqrt(true_cov)*z + r.normal(0, .05, n)
        y = np.sqrt(true_cov)*z + r.normal(0, .05, n)
        est.append(_cov2(x, y))
    emp_sd = np.std(est)
    r2 = np.random.default_rng(6)
    z = r2.normal(0,1,n); X = np.vstack([np.sqrt(true_cov)*z + r2.normal(0,.05,n) for _ in range(2)])
    reported = covariance_standard_errors(X)[0]
    print(f"  n={n:>4} empirical SD of cov = {emp_sd:.5f} ; one reported SE = {reported:.5f} "
          f"; ratio = {reported/emp_sd:.3f}   [n/(n-1)={n/(n-1):.3f}, sqrt={np.sqrt(n/(n-1)):.3f}]")

print()
print("EXP 8b: measurement variance is double counted in the SE")
print("="*72)
n, depth = 30, 10
r = np.random.default_rng(9)
p = np.full((2, n), 0.5)
counts = np.full((2, n), depth, float)
obs = r.binomial(depth, p) / depth
mv = beta_measurement_variance(obs, counts)
se_plain = covariance_standard_errors(obs)[0]
se_meas  = covariance_standard_errors_with_measurement(obs, mv)[0]
print(f"  empirical data already contains the sequencing noise.")
print(f"  SE without measurement term = {se_plain:.6f}")
print(f"  SE with    measurement term = {se_meas:.6f}   (inflation x{se_meas/se_plain:.2f})")

print()
print("="*72)
print("EXP 9: effect_probability / call_probability under post-selection")
print("="*72)
cfg = EigerConfig(calling_method="probabilistic", delta_threshold=0.20)
for n in (6, 10, 20, 60):
    r = np.random.default_rng(21)
    ep, mp, cp, sup = [], [], [], []
    for _ in range(2000):
        v = np.clip(0.5 + r.normal(0, 0.12, n), .001, .999)   # ONE group, no structure
        s = _probabilistic_separation_from_values(v, cfg)
        ep.append(s["effect_probability"]); mp.append(s["model_probability"])
        cp.append(s["call_probability"]); sup.append(s["call_supported"])
    print(f"  n={n:>3}  P(model_prob>0.8)={np.mean(np.array(mp)>0.8):.3f}"
          f"  mean effect_prob={np.nanmean(ep):.3f}"
          f"  P(call_prob>=0.8)={np.mean(np.array(cp)>=0.8):.3f}"
          f"  P(call_supported)={np.mean(sup):.3f}   <- all from a single Gaussian")
