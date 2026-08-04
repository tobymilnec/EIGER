import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/sessions/beautiful-upbeat-curie/mnt/EIGER/Python")
from eiger_core import EigerConfig, call_dmrs_from_matrix, estimate_covariance_threshold
from functions.processing import _cov2

def loc_df(n, chrom="chr1", step=2):
    s = np.arange(n) * step
    return pd.DataFrame({"chr": chrom, "start": s, "end": s + 1})

print("="*72)
print("EXP 13: the stated motivation - one weak edge inside a strong DMR")
print("="*72)
ncpg, n = 400, 30
ts, te = 150, 189
out = {"legacy": [], "probabilistic": []}
for rep in range(40):
    r = np.random.default_rng(900+rep)
    g = np.zeros(n); g[:n//2] = 1
    M = 0.5 + r.normal(0, 0.04, (ncpg, n))
    M[ts:te+1] += 0.28*g
    M[170] = 0.5 + r.normal(0, 0.04, n)      # one dropout CpG in the middle
    M = np.clip(M, .001, .999)
    for m in out:
        cfg = EigerConfig(calling_method=m, min_cpgs=4, covariance_threshold=0.012)
        core, _, _ = call_dmrs_from_matrix(M, loc_df(ncpg), config=cfg)
        piece = core[(core.cpg_end >= ts) & (core.cpg_start <= te)] if len(core) else core
        widths = (piece.cpg_end - piece.cpg_start + 1).tolist() if len(piece) else [0]
        out[m].append((len(piece), max(widths)))
for m, v in out.items():
    v = np.array(v, float)
    print(f"  {m:<14} fragments over the true DMR = {v[:,0].mean():.2f}   "
          f"largest recovered block = {v[:,1].mean():.1f}/40 CpGs")

print(); print("="*72)
print("EXP 14: latent factor - fixed threshold vs auto threshold false blocks")
print("="*72)
n, ncpg = 60, 5000
for load_sd in (0.04, 0.08, 0.15):
    r = np.random.default_rng(7)
    lat = r.normal(0, 1, n)
    load = r.normal(0, load_sd, (ncpg, 1))
    M = np.clip(0.5 + load*lat + r.normal(0, 0.05, (ncpg, n)), .001, .999)
    cov = np.array([_cov2(M[i], M[i+1]) for i in range(ncpg-1)])
    for tm, tv in (("fixed", 0.012), ("auto", None)):
        cfg = EigerConfig(calling_method="probabilistic", min_cpgs=4,
                          covariance_threshold=0.012,
                          covariance_threshold_method=tm)
        core, _, _ = call_dmrs_from_matrix(M, loc_df(ncpg), config=cfg)
        thr = core.attrs["covariance_threshold_used"]
        print(f"  loading sd={load_sd:<5} {tm:<6} thr={thr:.5f}  "
              f"false blocks on DMR-free data = {len(core):>4}  "
              f"(true 99.5th pct of edge cov = {np.quantile(cov,.995):.5f})")
    print()

print("="*72)
print("EXP 15: a distance-matched null keeps the latent factor but breaks locality")
print("="*72)
r = np.random.default_rng(7)
n, ncpg = 60, 5000
lat = r.normal(0, 1, n); load = r.normal(0, 0.08, (ncpg, 1))
M = np.clip(0.5 + load*lat + r.normal(0, 0.05, (ncpg, n)), .001, .999)
cov = np.array([_cov2(M[i], M[i+1]) for i in range(ncpg-1)])
elig = np.ones(ncpg-1, bool)
eig = estimate_covariance_threshold(M, elig, quantile=0.995, permutations=20,
                                    max_pairs=20000, seed=1).threshold
# proposed alternative: pair CpG i with CpG i+K (K large) on the same chromosome
alt = []
for K in (500, 1000, 2000):
    c = np.array([_cov2(M[i], M[i+K]) for i in range(0, ncpg-K)])
    alt.append(np.quantile(c, 0.995))
print(f"  true 99.5th pct of adjacent edge covariance (no DMRs) = {np.quantile(cov,.995):.5f}")
print(f"  EIGER per-sample resampling null threshold            = {eig:.5f}  "
      f"-> {np.mean(cov>=eig)*100:.1f}% of null edges pass")
for K, a in zip((500,1000,2000), alt):
    print(f"  distance-matched null, lag K={K:<5}                  = {a:.5f}  "
          f"-> {np.mean(cov>=a)*100:.1f}% of null edges pass")

print(); print("="*72)
print("EXP 16: partial read sharing at realistic WGBS depths")
print("="*72)
n, ncpg = 40, 2000
for depth in (5, 10, 30):
    for share in (0.0, 0.3, 0.6, 0.9):
        r = np.random.default_rng(11)
        cov = []
        for _ in range(ncpg):
            shared_n = int(depth*share); indep_n = depth - shared_n
            ms = r.binomial(shared_n, 0.5, n) if shared_n else np.zeros(n)
            m1 = ms + (r.binomial(indep_n, 0.5, n) if indep_n else 0)
            m2 = ms + (r.binomial(indep_n, 0.5, n) if indep_n else 0)
            cov.append(_cov2(m1/depth, m2/depth))
        cov = np.array(cov)
        print(f"  depth={depth:>3} shared fraction={share:<4} mean cov={cov.mean():.5f} "
              f"frac>=0.012: {np.mean(cov>=0.012):.3f}")
    print()
