"""
aei_radial_hist_NT.py
=====================
Histogram of the radial distribution (r/r_ISCO) of AEI-valid solutions
for the Novikov-Thorne model, coloured by disk zone (A, B, C).

Replicates the bottom-right panel of the 4-model radial distribution plot,
but as a standalone figure with the full (a, mdot, alpha) grid.

Usage:
    python aei_radial_hist_NT.py
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
from AEI_setups.nt_disc import disk_model_NT

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

MDOT_SCALE_NT = 10.0

A_GRID     = np.linspace(-0.99, 0.99, 19)
N_MDOT     = 30
N_ALPHA    = 20
MDOT_GRID  = np.logspace(-5, 0, N_MDOT)  * MDOT_SCALE_NT
ALPHA_GRID = np.logspace(-3, 0, N_ALPHA)

N_R        = 200
R_MAX_PHYS = 1000.0
K_MIN_WKB  = 1.0

ZONE_COLORS = {'A': '#3b82f6', 'B': '#f97316', 'C': '#22c55e'}

# ══════════════════════════════════════════════════════════════════════════════
# COLLECT VALID SOLUTIONS
# ══════════════════════════════════════════════════════════════════════════════

def collect_valid():
    """
    Collect r/r_ISCO and zone label for every valid AEI point
    over the full (a, mdot, alpha) grid.
    Returns arrays: r_risco_vals, zone_vals
    """
    r_risco_list = []
    zone_list    = []

    total = len(A_GRID) * N_MDOT * N_ALPHA
    count = 0

    R_RISCO_GRID = np.geomspace(1.0, R_MAX_PHYS, N_R)  # r/r_ISCO, starts at ISCO

    for a_val in A_GRID:
        isco_val = float(r_isco(a_val))
        r_vec    = R_RISCO_GRID * isco_val

        for mdot in MDOT_GRID:
            for alpha in ALPHA_GRID:
                count += 1
                if count % 2000 == 0:
                    print(f"  {count}/{total}  ({100*count/total:.0f}%)", end='\r')

                try:
                    result = disk_model_NT(r_vec, a_val, mdot,
                                           alpha_visc=alpha,
                                           hr=None, M=M_BH)
                except Exception:
                    continue

                B0_arr, Sigma_arr, cs_arr, hr_arr, zone_arr, _ = result

                k_arr = solve_k_aei(r_vec, a_val, B0_arr, Sigma_arr, cs_arr,
                                    m=mm, M=M_BH)

                mask = np.isfinite(k_arr) & (k_arr > 0)
                if not np.any(mask): continue

                mask &= (k_arr >= K_MIN_WKB) & (k_arr < 1.0 / np.maximum(hr_arr, 1e-10))
                if not np.any(mask): continue

                beta_arr = compute_beta(B0_arr, Sigma_arr, cs_arr,
                                        r_vec, hr_arr, M_BH)
                mask &= (beta_arr <= 1.0)
                if not np.any(mask): continue

                _B0i  = _make_interp(r_vec, B0_arr)
                _Sigi = _make_interp(r_vec, Sigma_arr)
                dQdr  = compute_dQdr(r_vec, a_val, _B0i, _Sigi, M_BH)
                mask &= (dQdr > 0)
                if not np.any(mask): continue

                r_risco_list.append(r_vec[mask] / isco_val)
                zone_list.append(zone_arr[mask])

    print()

    if not r_risco_list:
        return np.array([]), np.array([])
    return (np.concatenate(r_risco_list),
            np.concatenate(zone_list))


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_hist(r_risco_vals, zone_vals):
    fig, ax = plt.subplots(figsize=(8, 5))

    r_max_plot = np.percentile(r_risco_vals, 90) * 1.05
    bins = np.linspace(0, r_max_plot, 50)

    # stacked histogram by zone
    data_by_zone = []
    labels       = []
    colors       = []
    for zone in ['A', 'B', 'C']:
        mask = zone_vals == zone
        n = np.sum(mask)
        if n == 0:
            continue
        data_by_zone.append(r_risco_vals[mask])
        perc = 100*n/len(r_risco_vals)
        labels.append(f'Zone {zone} ({perc:.0f}%)')
        colors.append(ZONE_COLORS[zone])
    ax.hist(data_by_zone, bins=bins, stacked=True,
            color=colors, label=labels,
            alpha=0.9, edgecolor='none')

    # median line
    r_med = np.median(r_risco_vals)
    ax.axvline(r_med, color='black', ls='--', lw=1.5,
               label=f'median = {r_med:.0f}')

    ax.set_xlabel(r'$r\,/\,r_{\rm ISCO}$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    """
    ax.set_title(
        f'Radial distribution of AEI solutions  ($r/r_{{\\rm ISCO}}$)\n'
        f'Novikov-Thorne  —  N = {len(r_risco_vals):,} total valid points',
        fontsize=12
    )
    """
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('aei_radial_hist_NT.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('aei_radial_hist_NT.png', bbox_inches='tight', dpi=150)
    print("   -> saved aei_radial_hist_NT.pdf / .png")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  AEI radial histogram — Novikov-Thorne")
    print(f"  Grid: {len(A_GRID)} spins x {N_MDOT} mdot x {N_ALPHA} alpha")
    print(f"  = {len(A_GRID)*N_MDOT*N_ALPHA:,} combinations")
    print("=" * 65)

    print("\n> Collecting valid solutions...")
    r_risco_vals, zone_vals = collect_valid()
    print(f"   -> {len(r_risco_vals):,} valid points collected")

    if len(r_risco_vals) == 0:
        print("No valid solutions found.")
    else:
        for z in ['A', 'B', 'C']:
            n = np.sum(zone_vals == z)
            print(f"   Zone {z}: {n:,} ({100*n/len(zone_vals):.1f}%)")

        print("\n> Plotting...")
        plot_hist(r_risco_vals, zone_vals)
