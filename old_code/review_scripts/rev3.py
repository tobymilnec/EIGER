import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/sessions/beautiful-upbeat-curie/mnt/EIGER/Python")
from eiger_core import EigerConfig, call_dmrs_from_matrix, impute_for_covariance
from functions.processing import _cov2, covariance_standard_errors

def loc_df(n, chrom="chr1", step=2):
    s = np.arange(n) * step
    return pd.DataFrame({"chr": chrom, "start": s, "end": s + 1})

print("="*72); print("EXP 8a(v2): reported SE vs true sampling SD of the covariance"); print("="*72)
for n in (5, 8, 12, 20, 50, 150):
    r = np.random.default_rng(5)
    ests, ses = [], []
    for _ in range(3000):
        z = r.normal(0, 1, n)
        x = 0.14*z + r.normal(0, .05, n)
        y = 0.14*z + r.normal(0, .05, n)
        X = np.vstack([x, y])
        ests.append(_cov2(x, y)); ses.append(covariance_standard_errors(X)[0])
    emp = np.std(ests); rep = np.mean(ses)
    print(f"  n={n:>4}  true SD(cov)={emp:.5f}  mean reported SE={rep:.5f}  "
          f"bias={rep/emp:.3f}   median reported SE={np.median(ses):.5f}")

print(); print("="*72)
print("EXP 8c: is Phi((cov-thr)/SE) calibrated? coverage of the implied 1-sided test")
print("="*72)
from scipy.special import ndtr
for n in (6, 10, 20, 50, 200):
    r = np.random.default_rng(17)
    true_cov = 0.02
    p_at_truth = []
    for _ in range(4000):
        z = r.normal(0, 1, n)
        x = np.sqrt(true_cov)*z + r.normal(0, .05, n)
        y = np.sqrt(true_cov)*z + r.normal(0, .05, n)
        X = np.vstack([x, y])
        c = _cov2(x, y); se = covariance_standard_errors(X)[0]
        p_at_truth.append(ndtr((c - true_cov)/se) if se > 0 else 0.5)
    p_at_truth = np.array(p_at_truth)
    print(f"  n={n:>4}  P(support>0.975 when cov==threshold) = {np.mean(p_at_truth>0.975):.4f} "
          f"(nominal 0.025)   P(>0.5)={np.mean(p_at_truth>0.5):.3f} (nominal 0.5)")

print(); print("="*72)
print("EXP 10: HMM boundary behaviour - does smoothing over-extend blocks?")
print("="*72)
ncpg, n = 600, 40
true_start, true_end = 200, 229   # 30-CpG DMR
res = {"legacy": [], "probabilistic": []}
for rep in range(30):
    r = np.random.default_rng(500+rep)
    M = 0.5 + r.normal(0, 0.05, (ncpg, n))
    g = np.zeros(n); g[:n//2] = 1
    M[true_start:true_end+1] += 0.30*g
    M = np.clip(M, .001, .999)
    for m in res:
        cfg = EigerConfig(calling_method=m, min_cpgs=4, covariance_threshold=0.012)
        core, _, _ = call_dmrs_from_matrix(M, loc_df(ncpg), config=cfg)
        if core.empty: res[m].append((0,0,0,0)); continue
        # pick block with most overlap
        best = None
        for _, row in core.iterrows():
            a, b = int(row.cpg_start), int(row.cpg_end)
            ov = max(0, min(b, true_end) - max(a, true_start) + 1)
            if best is None or ov > best[0]: best = (ov, a, b)
        ov, a, b = best
        res[m].append((len(core), ov, a, b))
for m, v in res.items():
    v = np.array(v, float)
    print(f"  {m:<14} blocks/rep={v[:,0].mean():.2f}  recovered CpGs={v[:,1].mean():.1f}/30  "
          f"mean start={v[:,2].mean():.1f} (true 200)  mean end={v[:,3].mean():.1f} (true 229)  "
          f"mean width={(v[:,3]-v[:,2]+1).mean():.1f}")

print(); print("="*72)
print("EXP 11: mean imputation deflates covariance in proportion to missingness")
print("="*72)
n, reps = 40, 400
for miss in (0.0, 0.1, 0.2, 0.4):
    r = np.random.default_rng(31)
    cs = []
    for _ in range(reps):
        g = np.zeros(n); g[:n//2] = 1
        x = 0.5 + 0.30*g + r.normal(0, .05, n)
        y = 0.5 + 0.30*g + r.normal(0, .05, n)
        M = np.vstack([x, y])
        M[r.random(M.shape) < miss] = np.nan
        cs.append(_cov2(*impute_for_covariance(M)))
    print(f"  missing={miss:<5} mean adjacent cov = {np.mean(cs):.5f}  "
          f"(relative to complete data: {np.mean(cs)/0.0225:.2f}x, expected (1-m)^2={(1-miss)**2:.2f})")

print(); print("="*72)
print("EXP 12: auto threshold on iid null (is the 0.995 quantile calibrated?)")
print("="*72)
for n in (10, 40, 100):
    r = np.random.default_rng(77)
    M = np.clip(0.5 + r.normal(0, 0.10, (4000, n)), .001, .999)
    cfg = EigerConfig(covariance_threshold_method="auto", calling_method="probabilistic", min_cpgs=4)
    core, _, _ = call_dmrs_from_matrix(M, loc_df(4000), config=cfg)
    cov = np.array([_cov2(M[i], M[i+1]) for i in range(3999)])
    thr = core.attrs["covariance_threshold_used"]
    print(f"  n={n:>4}  auto thr={thr:.5f}  true 99.5th pct of null edges={np.quantile(cov,.995):.5f}"
          f"  frac edges passing={np.mean(cov>=thr):.4f}  blocks called={len(core)}")
