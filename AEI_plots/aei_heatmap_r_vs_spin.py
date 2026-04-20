"""
aei_heatmap_r_vs_spin.py
========================
Heatmap of AEI solutions (all checks: WKB + beta + shear) in the
(spin a, r/r_ISCO) plane, coloured by the median dimensionless
wavenumber k.

Grid structure — same philosophy as the (B00, Sigma0) heatmap:
  - spin axis   : A_GRID (e.g. 60 values), one column per spin
  - radius axis : R_RISCO_GRID, a fixed log-spaced grid in r/r_ISCO
                  shared by all spin values (no binning)

For each cell (a_i, r_j/r_ISCO) the colour is the median k over all
(B00, Sigma0) pairs that yield a valid AEI solution at that exact
(spin, normalised radius). Cells with no valid solution are blank.

Overlaid curves (smooth, 500-point spin grid):
  - ISCO : r/r_ISCO = 1  (horizontal reference line)
  - ILR  : r_ILR(a) / r_ISCO(a)
  - OLR  : r_OLR(a) / r_ISCO(a)
  - CR   : r_CR(a)  / r_ISCO(a)

Usage:
    python aei_heatmap_r_vs_spin.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')

from setup import M_BH, NU0, r_isco
from AEI_setups.aei_common import (
    solve_k_aei, compute_beta, compute_dQdr,
    r_ilr, r_olr, r_corotation,
    HOR, mm, _make_interp,
)
from aei_2.simple_disc import disk_model_simple

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

ALP_B_v1 = 5/4
ALP_S_v1 = 3/5

# Spin grid — one column per value, no binning
A_GRID = np.linspace(-0.99, 0.99, 60)

# (B00, Sigma0) grid — aggregated over for each cell
N_B00    = 50
N_SIGMA0 = 45
B00_GRID    = np.logspace(1, 8, N_B00)
SIGMA0_GRID = np.logspace(2, 7, N_SIGMA0)

# Shared r/r_ISCO grid — one row per value, no binning
# Starts slightly below ISCO so the lower boundary is visible in the plot.
# Upper end covers up to R_MAX_PHYS rg; the plot y-limit is set from data.
N_R              = 300
R_RISCO_MIN      = 0.9      # slightly below ISCO to show the lower boundary
R_MAX_PHYS       = 1000.0   # maximum physical radius [r_g] to search
R_RISCO_MAX_GRID = 500  # generous upper bound for the grid
R_RISCO_GRID     = np.geomspace(R_RISCO_MIN, R_RISCO_MAX_GRID, N_R)

# WKB range
K_MIN_WKB = 1.0
K_MAX_WKB = 1.0 / HOR   # = 20 for HOR = 0.05

# Fine spin grid for smooth resonance curves
A_CURVE = np.linspace(-0.99, 0.99, 500)

# ══════════════════════════════════════════════════════════════════════════════
# DISK MODEL
# ══════════════════════════════════════════════════════════════════════════════

def disk_model_v1(r_rg, a, B00, Sigma0):
    return disk_model_simple(r_rg, a, B00, Sigma0,
                             alpha_B=ALP_B_v1, alpha_S=ALP_S_v1,
                             hr=HOR, M=M_BH)

# ══════════════════════════════════════════════════════════════════════════════
# SCAN
# ══════════════════════════════════════════════════════════════════════════════

def build_heatmap():
    """
    For each (a_i, r_j/r_ISCO) cell accumulate k values over all
    (B00, Sigma0) combinations that pass all checks at that exact
    normalised radius. Return median-k map of shape (N_A, N_R).
    """
    N_A = len(A_GRID)
    # k_accum[i][j] = list of valid k at spin i, r/r_ISCO index j
    k_accum = [[[] for _ in range(N_R)] for _ in range(N_A)]

    total = N_A * N_B00 * N_SIGMA0
    count = 0

    for i, a_val in enumerate(A_GRID):
        isco_val = float(r_isco(a_val))
        # physical radii corresponding to the shared r/r_ISCO grid
        r_vec = R_RISCO_GRID * isco_val
        # only radii >= r_ISCO are physical
        phys_mask    = R_RISCO_GRID >= 1.0
        r_vec_phys   = r_vec[phys_mask]
        phys_indices = np.where(phys_mask)[0]

        for B00 in B00_GRID:
            for Sigma0 in SIGMA0_GRID:
                count += 1
                if count % 5000 == 0:
                    print(f"  {count}/{total}  ({100*count/total:.0f}%)", end='\r')

                result = disk_model_v1(r_vec_phys, a_val, B00, Sigma0)
                B0_arr, Sigma_arr, cs_arr, hr_arr, _, _ = result

                k_arr = solve_k_aei(r_vec_phys, a_val, B0_arr, Sigma_arr, cs_arr,
                                    m=mm, M=M_BH)

                mask = np.isfinite(k_arr) & (k_arr > 0)
                if not np.any(mask): continue

                mask &= (k_arr >= K_MIN_WKB) & (k_arr <= K_MAX_WKB)
                if not np.any(mask): continue

                beta_arr = compute_beta(B0_arr, Sigma_arr, cs_arr,
                                        r_vec_phys, HOR, M_BH)
                mask &= (beta_arr <= 1.0)
                if not np.any(mask): continue

                _B0i  = _make_interp(r_vec_phys, B0_arr)
                _Sigi = _make_interp(r_vec_phys, Sigma_arr)
                dQdr  = compute_dQdr(r_vec_phys, a_val, _B0i, _Sigi, M_BH)
                mask &= (dQdr > 0)
                if not np.any(mask): continue

                # deposit into the correct global indices
                for local_j in np.where(mask)[0]:
                    global_j = phys_indices[local_j]
                    k_accum[i][global_j].append(k_arr[local_j])

    print()

    k_map = np.full((N_A, N_R), np.nan)
    for i in range(N_A):
        for j in range(N_R):
            if k_accum[i][j]:
                k_map[i, j] = np.median(k_accum[i][j])

    return k_map

# ══════════════════════════════════════════════════════════════════════════════
# RESONANCE CURVES
# ══════════════════════════════════════════════════════════════════════════════

def compute_resonance_curves():
    print("  Computing resonance curves...", end=' ')
    curves = {name: np.full(len(A_CURVE), np.nan)
              for name in ('ILR', 'OLR', 'CR')}
    for i, a_val in enumerate(A_CURVE):
        isco = float(r_isco(a_val))
        ilr  = r_ilr(a_val)
        olr  = r_olr(a_val)
        cr   = r_corotation(a_val)
        if np.isfinite(ilr): curves['ILR'][i] = ilr / isco
        if np.isfinite(olr): curves['OLR'][i] = olr / isco
        if np.isfinite(cr):  curves['CR'][i]  = cr  / isco
    print("done")
    return curves

# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(k_map, curves):
    fig, ax = plt.subplots(figsize=(10, 5))

    valid_k = k_map[np.isfinite(k_map)]
    if valid_k.size == 0:
        print("WARNING: no valid solutions to plot.")
        return

    norm = LogNorm(vmin=valid_k.min(), vmax=valid_k.max())
    cmap = plt.cm.viridis

    # pcolormesh: x = spin, y = r/r_ISCO
    # k_map shape (N_A, N_R) → transpose to (N_R, N_A) for (y, x) convention
    im = ax.pcolormesh(
        A_GRID, R_RISCO_GRID, k_map.T,
        norm=norm, cmap=cmap, shading='auto',
    )
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'median $k$  (dimensionless)', fontsize=11)

    ax.set_yscale('log')

    # ISCO
    ax.axhline(1.0, color='white', lw=2.0, ls='-',
               label=r'ISCO  ($r/r_{\rm ISCO}=1$)', zorder=5)

    # resonance curves
    style = {
        'ILR': dict(color='dodgerblue', lw=1.8, ls='--',
                    label=r'$r_{\rm ILR}/r_{\rm ISCO}$'),
        'OLR': dict(color='magenta',     lw=2.2, ls='--',
                    label=r'$r_{\rm OLR}/r_{\rm ISCO}$'),
        'CR':  dict(color='gold',       lw=1.8, ls=':',
                    label=r'$r_{\rm CR}/r_{\rm ISCO}$'),
    }
    for name, arr in curves.items():
        mask = np.isfinite(arr)
        if mask.any():
            ax.plot(A_CURVE[mask], arr[mask], zorder=6, **style[name])

    ax.set_xlabel(r'Spin  $a$', fontsize=12)
    ax.set_ylabel(r'$r \,/\, r_{\rm ISCO}$', fontsize=12)
    ax.set_xlim(A_GRID[0] - 0.02, A_GRID[-1] + 0.02)
    # upper y limit: highest r/r_ISCO row with any valid solution + 20% margin
    valid_rows = np.where(np.any(np.isfinite(k_map), axis=0))[0]
    r_max_data = R_RISCO_GRID[valid_rows[-1]] * 1.2 if valid_rows.size else R_RISCO_MAX_GRID
    ax.set_ylim(R_RISCO_MIN, r_max_data)
    ax.grid(True, which='both', alpha=0.15, lw=0.5)
    ax.legend(fontsize=9, loc='upper left')
    """
    ax.set_title(
        r"AEI solutions — Simple-v1  ($\alpha_B=5/4$, $\alpha_\Sigma=3/5$)"
        "\n"
        r"All checks (WKB + $\beta$ + shear) — colour = median $k$",
        fontsize=12
    )
    """
    plt.tight_layout()
    plt.savefig('aei_heatmap_r_vs_spin.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('aei_heatmap_r_vs_spin.png', bbox_inches='tight', dpi=150)
    print("   -> saved aei_heatmap_r_vs_spin.pdf / .png")
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  AEI Heatmap: r/r_ISCO vs spin — Simple-v1")
    print(f"  Spin grid  : {len(A_GRID)} values in [{A_GRID[0]:.1f}, {A_GRID[-1]:.1f}]")
    print(f"  Radius grid: {N_R} points in r/r_ISCO in [{R_RISCO_MIN}, {R_RISCO_MAX_GRID}]")
    print(f"  (B00, Sigma0): {N_B00} x {N_SIGMA0} = {N_B00*N_SIGMA0} cells")
    print(f"  WKB: {K_MIN_WKB:.0f} <= k <= {K_MAX_WKB:.0f}  |  all checks active")
    print("=" * 65)

    print("\n> Building heatmap...")
    k_map = build_heatmap()
    n_filled = np.sum(np.isfinite(k_map))
    print(f"   -> {n_filled} filled cells out of {k_map.size} ({100*n_filled/k_map.size:.1f}%)")

    curves = compute_resonance_curves()

    print("\n> Plotting...")
    plot_heatmap(k_map, curves)
