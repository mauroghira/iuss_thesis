"""
aei_NT_ISCO_analysis.py
=======================
For the Novikov-Thorne model, for every (a, mdot, alpha) combination
that yields at least one valid AEI solution:
  - computes B_ISCO and Sigma_ISCO via disk_inner_values_NT
  - collects the median physical radius r [r_g] of valid solutions

Produces:
  1. LaTeX table: r_5%, B_ISCO_5%, Sigma_ISCO_50%  for each H/r value
     (same structure as the Simple-model table)
  2. Scatter plot of valid solutions in (B_ISCO, Sigma_ISCO) space,
     coloured by median r [r_g]

Usage:
    python aei_NT_ISCO_analysis.py
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
    ALPHA_VISC, mm, _make_interp,
)
from AEI_setups.nt_disc import disk_model_NT, disk_inner_values_NT

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

MDOT_SCALE_NT = 10.0

A_GRID     = np.linspace(-0.99, 0.99, 19)
N_MDOT     = 30
N_ALPHA    = 30
MDOT_GRID  = np.logspace(-5, 0, N_MDOT)
ALPHA_GRID = np.logspace(-3, 0, N_ALPHA)


N_R        = 200
R_MAX_PHYS = 1000.0
K_MIN_WKB  = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# CORE SCAN
# ══════════════════════════════════════════════════════════════════════════════

def collect_valid_with_isco():
    """
    For each (a, mdot, alpha) with at least one valid AEI solution,
    collect:
      r_list      : physical radii [r_g] of valid points
      B_ISCO_list : B at ISCO for that (a, mdot, alpha)
      S_ISCO_list : Sigma at ISCO for that (a, mdot, alpha)
    H/r is taken directly from the NT model (hr_arr), not fixed.
    Returns arrays of same length (one entry per valid radial point).
    """
    R_RISCO_GRID = np.geomspace(1.0, R_MAX_PHYS, N_R)

    r_list, B_list, S_list, hr_list = [], [], [], []

    # cache B_ISCO and Sigma_ISCO — expensive, call once per (a, mdot, alpha)
    isco_cache = {}

    total = len(A_GRID) * N_MDOT * N_ALPHA
    count = 0

    for a_val in A_GRID:
        isco_val = float(r_isco(a_val))
        r_vec    = R_RISCO_GRID * isco_val

        for mdot in MDOT_GRID:
            mdot = mdot * MDOT_SCALE_NT
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

                B0_arr, Sigma_arr, cs_arr, hr_arr, _, _ = result

                k_arr = solve_k_aei(r_vec, a_val, B0_arr, Sigma_arr, cs_arr,
                                    m=mm, M=M_BH)

                mask = np.isfinite(k_arr) & (k_arr > 0)
                if not np.any(mask): continue

                mask &= (k_arr >= K_MIN_WKB) & (k_arr < 1.0 / hr_arr)
                if not np.any(mask): continue

                beta_arr = compute_beta(B0_arr, Sigma_arr, cs_arr,
                                        r_vec, hr_arr, M_BH)
                mask &= (beta_arr <= 1.0)
                if not np.any(mask): 
                    print("no beta")
                    continue

                _B0i  = _make_interp(r_vec, B0_arr)
                _Sigi = _make_interp(r_vec, Sigma_arr)
                dQdr  = compute_dQdr(r_vec, a_val, _B0i, _Sigi, M_BH)
                mask &= (dQdr > 0)
                if not np.any(mask): 
                    print("no shear")
                    continue

                # get B_ISCO and Sigma_ISCO (cached)
                key = (round(a_val, 6), round(mdot, 12), round(alpha, 8))
                if key not in isco_cache:
                    try:
                        iv = disk_inner_values_NT(a_val, mdot,
                                                   alpha_visc=alpha, M=M_BH)
                        isco_cache[key] = (iv['B_ISCO'], iv['Sigma_ISCO'])
                    except Exception:
                        isco_cache[key] = (np.nan, np.nan)

                B_isco, S_isco = isco_cache[key]

                if not np.isfinite(B_isco):
                    print("no b")
                if not np.isfinite(S_isco):
                    print("no s")
                if not (np.isfinite(B_isco) and np.isfinite(S_isco)):
                    print("no bs")
                    continue

                n_valid = np.sum(mask)
                r_list.append(r_vec[mask])
                hr_list.append(hr_arr[mask])
                B_list.append(np.full(n_valid, B_isco))
                S_list.append(np.full(n_valid, S_isco))

    print()
    if not r_list:
        print("no r")
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (np.concatenate(r_list),
            np.concatenate(B_list),
            np.concatenate(S_list),
            np.concatenate(hr_list))


# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLE
# ══════════════════════════════════════════════════════════════════════════════

def sci(x, d=1):
    if not np.isfinite(x): return r'---'
    e = int(np.floor(np.log10(abs(x))))
    c = x / 10**e
    return rf'${c:.{d}f}\times10^{{{e}}}$'

def build_table(rows):
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{Statistics of AEI-valid solutions for the Novikov-Thorne model over the full $(a,\,\dot{m},\,\alpha)$ grid. 5\% indicates the 5th-percentile.}')
    lines.append(r'\label{tab4:NT_stats}')
    lines.append(r'\begin{tabular}{cccc}')
    lines.append(r'\toprule')
    lines.append(r'$r^{5\%}$ [r$_g$] & $B_{{\rm ISCO}}^{5\%}$ [G] & $\langle\Sigma_{0}\rangle}$ [g\,cm$^{-2}$] & $(H/r)^{max}$ \\')
    lines.append(r'\midrule')
    for row in rows:
        lines.append(
            rf"{sci(row['r_p5'])} & {sci(row['B_p5'])} & {sci(row['S_med'])} & {sci(row['hr_max'])}\\"
        )
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SCATTER PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(hr_arr, B_arr, S_arr):
    """Scatter plot in (B_ISCO, Sigma_ISCO) space, coloured by r [r_g]."""
    fig, ax = plt.subplots(figsize=(8, 6))

    hr_min = hr_arr.min()
    hr_max = hr_arr.max()
    if hr_max / hr_min > 1000:
        norm = LogNorm(vmin=hr_min, vmax=hr_max)
    else:
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=hr_min, vmax=hr_max)

    sc = ax.scatter(S_arr, B_arr,
                    c=hr_arr, norm=norm, cmap='plasma',
                    s=3, alpha=0.4, rasterized=True)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r'$r$  [r$_g$]', fontsize=11)
    ax.plot(1e4, 1e4, 'x', color='green', ms=10, mew=2, zorder=5, label="Reference values")
    ax.axhline(1e4, color='green', lw=1.8, ls='--',)
    ax.legend(fontsize=8, loc='lower left')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$B_{\rm ISCO}$  [G]', fontsize=12)
    ax.set_xlabel(r'$\Sigma_{0}$  [g cm$^{-2}$]', fontsize=12)
    """
    ax.set_title(
        r'AEI solutions — Novikov-Thorne  ($H/r$ from model)'
        '\n'
        r'$(B_{\rm ISCO},\,\Sigma_{\rm ISCO})$ space — colour = $r$ [r$_g$]'
        f'\nN = {len(r_arr):,} valid points',
        fontsize=11
    )
    """
    ax.grid(True, which='both', alpha=0.15)

    plt.tight_layout()
    plt.savefig('aei_NT_BISOC_SigmaISCO.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('aei_NT_BISOC_SigmaISCO.png', bbox_inches='tight', dpi=150)
    print("   -> saved aei_NT_BISOC_SigmaISCO.pdf / .png")
    plt.show()

# --------------------------

def plot_heatmap(hr_arr, B_arr, S_arr, n_bins=60):
    """
    Heatmap in (Sigma_ISCO, B_ISCO) space.
 
    Each cell shows the median H/r of the AEI-valid solutions that fall
    inside it.  The colour scale runs from the global min to the global max
    of hr_arr (no percentile clipping).
 
    Parameters
    ----------
    hr_arr : array-like   H/r values  (physical NT, one per valid point)
    B_arr  : array-like   B_ISCO  [G]
    S_arr  : array-like   Sigma_ISCO  [g/cm^2]
    n_bins : int          number of bins along each axis (default 60)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.stats import binned_statistic_2d
 
    hr_arr = np.asarray(hr_arr, float)
    B_arr  = np.asarray(B_arr,  float)
    S_arr  = np.asarray(S_arr,  float)
 
    # --- log-space bin edges on both axes -----------------------------------
    S_edges = np.geomspace(S_arr.min(), S_arr.max(), n_bins + 1)
    B_edges = np.geomspace(B_arr.min(), B_arr.max(), n_bins + 1)
 
    # --- median H/r in each cell --------------------------------------------
    stat, _, _, _ = binned_statistic_2d(
        S_arr, B_arr, hr_arr,
        statistic='median',
        bins=[S_edges, B_edges],
    )
    # stat shape: (n_bins_S, n_bins_B) — transpose so B is on y-axis
 
    # --- colour scale: true min/max of hr_arr (no clipping) -----------------
    hr_min = hr_arr.min()
    hr_max = hr_arr.max()
 
    # Use LogNorm if the range spans more than one decade, else linear
    if hr_max / hr_min > 10:
        norm = LogNorm(vmin=hr_min, vmax=hr_max)
    else:
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=hr_min, vmax=hr_max)
 
    # --- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
 
    # pcolormesh needs edges (n+1 points) on each axis
    im = ax.pcolormesh(
        S_edges, B_edges, stat.T,          # transpose: rows=B, cols=S
        norm=norm,
        cmap='plasma',
        shading='flat',
        rasterized=True,
    )
 
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    #cb.set_label(r'median $H/r$', fontsize=11)
    cb.set_label(r'$r$  [r$_g$]', fontsize=11)


    # --- reference marker and line ------------------------------------------
    ax.plot(1e4, 1e4, 'x', color='green', ms=10, mew=2,
            zorder=5, label='Reference values')
    ax.axhline(1e4, color='green', lw=1.8, ls='--')
    ax.legend(fontsize=8, loc='lower left')
 
    # --- count annotation ---------------------------------------------------
    n_cells = int(np.sum(np.isfinite(stat)))
    ax.text(0.98, 0.02,
            f'N points = {len(hr_arr):,}\nN cells = {n_cells:,}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.45))
 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$B_{\rm ISCO}$  [G]',              fontsize=12)
    ax.set_xlabel(r'$\Sigma_{0}$  [g cm$^{-2}$]', fontsize=12)
    ax.grid(True, which='both', alpha=0.15)
 
    plt.tight_layout()
    plt.savefig('aei_NT_BISCO_SigmaISCO_heatmap.pdf',
                bbox_inches='tight', dpi=150)
    plt.savefig('aei_NT_BISCO_SigmaISCO_heatmap.png',
                bbox_inches='tight', dpi=150)
    print('   -> saved aei_NT_BISCO_SigmaISCO_heatmap.pdf / .png')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  AEI NT — ISCO analysis (table + scatter)")
    print(f"  Grid: {len(A_GRID)} a x {N_MDOT} mdot x {N_ALPHA} alpha")
    print("  H/r: taken from NT model (per-point hr_arr)")
    print("=" * 65)

    # ── single scan (H/r from NT model) ────────────────────────────────────
    print("\n> Scanning (H/r from NT model)...")
    r_arr, B_arr, S_arr, hr_arr = collect_valid_with_isco()
    table_rows = []
    if r_arr.size == 0:
        print("   -> no valid solutions")
    else:
        table_rows.append(dict(
            hor='NT model',
            r_p5 = np.percentile(r_arr,  5),
            B_p5 = np.percentile(B_arr,  5),
            S_med= np.percentile(S_arr, 50),
            hr_max=np.max(hr_arr),
            n    = r_arr.size,
        ))
        print(f"   -> N={r_arr.size:,}  r_p5={table_rows[-1]['r_p5']:.2f}  "
              f"B_p5={table_rows[-1]['B_p5']:.2e}  S_med={table_rows[-1]['S_med']:.2e}")

    table = build_table(table_rows)
    print("\n" + "=" * 65)
    print(table)
    with open('aei_NT_table.tex', 'w') as f:
        f.write(table + '\n')
    print("   -> saved aei_NT_table.tex")

    # ── scatter plot (reuses data from scan above) ─────────────────────────
    if r_arr.size > 0:
        print("\n> Plotting scatter...")
        #plot_scatter(r_arr, B_arr, S_arr)
        plot_heatmap(r_arr, B_arr, S_arr)
    else:
        print("   -> no valid solutions for scatter plot")
