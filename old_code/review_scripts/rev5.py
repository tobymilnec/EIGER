import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "/sessions/beautiful-upbeat-curie/mnt/EIGER/Python")
from eiger_core import (EigerConfig, call_dmrs_from_matrix, covariance_edge_support,
                        smooth_edge_support, _probabilistic_separation_from_values)
from functions.processing import _cov2, covariance_blocks, covariance_standard_errors

def loc_df(n, chrom="chr1", step=2):
    s = np.arange(n)*step
    return pd.DataFrame({"chr": chrom, "start": s, "end": s+1})

print("="*72); print("EXP 17: can the HMM bridge ONE weak edge? (single attenuated CpG)"); print("="*72)
ncpg, n = 400, 30; ts, te = 150, 189
for atten in (1.0, 0.6, 0.4, 0.2, 0.0):
    res = {}
    for m in ("legacy", "probabilistic"):
        frag, width = [], []
        for rep in range(25):
            r = np.random.default_rng(700+rep)
            g = np.zeros(n); g[:n//2] = 1
            M = 0.5 + r.normal(0, 0.04, (ncpg, n))
            M[ts:te+1] += 0.28*g
            M[170] = 0.5 + atten*0.28*g + r.normal(0, 0.04, n)
            M = np.clip(M, .001, .999)
            cfg = EigerConfig(calling_method=m, min_cpgs=4, covariance_threshold=0.012)
            core, _, _ = call_dmrs_from_matrix(M, loc_df(ncpg), config=cfg)
            p = core[(core.cpg_end >= ts) & (core.cpg_start <= te)] if len(core) else core
            frag.append(len(p)); width.append(max((p.cpg_end-p.cpg_start+1).tolist() or [0]) if len(p) else 0)
        res[m] = (np.mean(frag), np.mean(width))
    print(f"  attenuation of CpG170 to {atten:>4.0%} of the effect: "
          f"legacy {res['legacy'][0]:.2f} frags / {res['legacy'][1]:.1f} CpGs   |   "
          f"probabilistic {res['probabilistic'][0]:.2f} frags / {res['probabilistic'][1]:.1f} CpGs")

print(); print("="*72); print("EXP 18: HMM sensitivity to hmm_signal_prior (an unvalidated free parameter)")
print("="*72)
r = np.random.default_rng(3)
g = np.zeros(30); g[:15] = 1
M = 0.5 + r.normal(0, 0.04, (400, 30)); M[150:190] += 0.28*g
M[170] = 0.5 + 0.4*0.28*g + r.normal(0, 0.04, 30)
M = np.clip(M, .001, .999)
for prior in (0.01, 0.05, 0.2, 0.5, 0.8):
    for pers in (0.7, 0.9, 0.99):
        cfg = EigerConfig(calling_method="probabilistic", hmm_signal_prior=prior,
                          hmm_signal_persistence=pers, min_cpgs=4)
        core, _, _ = call_dmrs_from_matrix(M, loc_df(400), config=cfg)
        print(f"    prior={prior:<5} persistence={pers:<5} -> {len(core)} blocks, "
              f"widths={sorted((core.cpg_end-core.cpg_start+1).tolist(), reverse=True)[:3] if len(core) else []}")

print(); print("="*72); print("EXP 19: runtime / memory of the hot paths at chromosome scale"); print("="*72)
n = 60
for ncpg in (200_000, 1_000_000):
    M = np.random.default_rng(0).normal(0.5, 0.1, (ncpg, n))
    t=time.time(); cov = covariance_blocks(M); t_cov=time.time()-t
    print(f"  {ncpg:>9,} CpGs x {n}: covariance_blocks {t_cov:6.2f}s  "
          f"(returns a Python list of {len(cov):,} floats ~ {len(cov)*32/1e6:.0f} MB)")
    cov = np.asarray(cov); t=time.time(); se = covariance_standard_errors(M); t_se=time.time()-t
    sup = covariance_edge_support(cov, se, 0.012)
    elig = np.ones(ncpg-1, bool)
    t=time.time(); post = smooth_edge_support(sup, elig); t_hmm=time.time()-t
    print(f"                          covariance_SEs {t_se:6.2f}s   "
          f"smooth_edge_support (pure Python loop) {t_hmm:6.2f}s")
    del M
print("  -> extrapolated to ~28M autosomal CpGs, smoothing alone costs "
      f"~{t_hmm*28:.0f}s per run (x2 for the auto two-pass, x B for bootstraps)")
