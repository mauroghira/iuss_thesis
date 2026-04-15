"""
aei_k_vs_hor.py
===============
Plot of median k (and its IQR band) as a function of H/r,
for Simple-v1 and Simple-v2, on the same axes.

For each (model, H/r) combination the full (a, B00, Sigma0) grid is
scanned with all checks active; k_median, k_p25 and k_p75 are computed
over all valid solutions.

Usage:
    python aei_k_vs_hor.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')

from setup import M_BH, NU0, r_isco
from AEI_setups.aei_common import (
    solve_k_aei, compute_beta, compute_dQdr,
    mm, _make_interp,
)
from AEI_setups.simple_disc import disk_model_simple

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    'Simple-v1': dict(alpha_B=5/4, alpha_S=3/5, color='#6366f1', ls='-'),
    'Simple-v2': dict(alpha_B=1.7,  alpha_S=3/5, color='#f97316', ls='--'),
}

# Dense H/r axis — log-spaced for a smooth curve
HOR_GRID = np.logspace(-3, -1, 20)   # 0.001 → 0.1, 20 points

A_GRID      = np.linspace(-0.9, 0.9, 19)
N_B00       = 50
N_SIGMA0    = 45
B00_GRID    = np.logspace(1, 8, N_B00)
SIGMA0_GRID = np.logspace(2, 7, N_SIGMA0)

N_R        = 300
R_MAX_PHYS = 1000.0
K_MIN_WKB  = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# SCAN — collect all valid k values for one (model, H/r)
# ══════════════════════════════════════════════════════════════════════════════

def collect_k(alpha_B, alpha_S, hor):
    k_max_wkb    = 1.0 / hor
    R_RISCO_GRID = np.geomspace(1.0, R_MAX_PHYS, N_R)
    k_list       = []

    for a_val in A_GRID:
        isco_val         = float(r_isco(a_val))
        r_risco_max_spin = R_MAX_PHYS / isco_val
        rr               = R_RISCO_GRID[R_RISCO_GRID <= r_risco_max_spin]
        r_vec            = rr * isco_val

        for B00 in B00_GRID:
            for Sigma0 in SIGMA0_GRID:

                result = disk_model_simple(r_vec, a_val, B00, Sigma0,
                                           alpha_B=alpha_B, alpha_S=alpha_S,
                                           hr=hor, M=M_BH)
                B0_arr, Sigma_arr, cs_arr, hr_arr, _, _ = result

                k_arr = solve_k_aei(r_vec, a_val, B0_arr, Sigma_arr, cs_arr,
                                    m=mm, M=M_BH)

                mask = np.isfinite(k_arr) & (k_arr > 0)
                if not np.any(mask): continue

                mask &= (k_arr >= K_MIN_WKB) & (k_arr <= k_max_wkb)
                if not np.any(mask): continue

                beta_arr = compute_beta(B0_arr, Sigma_arr, cs_arr,
                                        r_vec, hor, M_BH)
                mask &= (beta_arr <= 1.0)
                if not np.any(mask): continue

                _B0i  = _make_interp(r_vec, B0_arr)
                _Sigi = _make_interp(r_vec, Sigma_arr)
                dQdr  = compute_dQdr(r_vec, a_val, _B0i, _Sigi, M_BH)
                mask &= (dQdr > 0)
                if not np.any(mask): continue

                k_list.append(k_arr[mask])

    if not k_list:
        return np.array([])
    return np.concatenate(k_list)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  AEI median k vs H/r — Simple-v1 & v2")
    print(f"  H/r grid : {len(HOR_GRID)} points in [{HOR_GRID[0]:.3f}, {HOR_GRID[-1]:.3f}]")
    print(f"  Grid     : {len(A_GRID)} spins x {N_B00} B00 x {N_SIGMA0} Sigma0")
    print("=" * 65)

    results = {label: {'hor': [], 'k_med': [], 'k_p25': [], 'k_p75': [], 'k_min': [], 'n': []}
               for label in MODELS}

    total_scans = len(MODELS) * len(HOR_GRID)
    scan_count  = 0

    for label, cfg in MODELS.items():
        for hor in HOR_GRID:
            scan_count += 1
            print(f"\n[{scan_count}/{total_scans}] {label}  H/r={hor:.4f}  "
                  f"(k_max={1/hor:.1f})")

            k_arr = collect_k(cfg['alpha_B'], cfg['alpha_S'], hor)

            if k_arr.size == 0:
                print("   -> no valid solutions")
                continue

            k_med = np.median(k_arr)
            k_p25 = np.percentile(k_arr, 25)
            k_p75 = np.percentile(k_arr, 75)
            k_min = np.min(k_arr)
            print(f"   -> N={k_arr.size:,}  k_med={k_med:.2f}  "
                  f"IQR=[{k_p25:.2f}, {k_p75:.2f}]")

            d = results[label]
            d['hor'].append(hor)
            d['k_med'].append(k_med)
            d['k_p25'].append(k_p25)
            d['k_p75'].append(k_p75)
            d['k_min'].append(k_min)
            d['n'].append(k_arr.size)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3.5))

    for label, cfg in MODELS.items():
        d = results[label]
        if not d['hor']:
            continue
        hor_arr = np.array(d['hor'])
        k_med   = np.array(d['k_med'])
        k_p25   = np.array(d['k_p25'])
        k_p75   = np.array(d['k_p75'])
        k_min   = np.array(d['k_min'])

        ax.plot(hor_arr, k_med, color=cfg['color'], ls=cfg['ls'],
                lw=2.2, marker='o', ms=5, label=label)
        ax.fill_between(hor_arr, k_p25, k_p75,
                        color=cfg['color'], alpha=0.18)
        #ax.plot(hor_arr, k_min, color="blue", ls="-",
        #        lw=2.2, marker='o', ms=5, label=label)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$H/R$', fontsize=13)
    ax.set_ylabel(r'median $k$  (dimensionless)', fontsize=13)
    """
    ax.set_title(
        r'Median $k$ of AEI-valid solutions vs $H/r$' '\n'
        r'All checks (WKB + $\beta$ + shear) — shaded band: IQR',
        fontsize=12
    )
    """
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.2)

    plt.tight_layout()
    plt.savefig('aei_k_vs_hor.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('aei_k_vs_hor.png', bbox_inches='tight', dpi=150)
    print("\n   -> saved aei_k_vs_hor.pdf / .png")
    plt.show()
