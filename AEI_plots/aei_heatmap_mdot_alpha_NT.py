"""
aei_heatmap_mdot_alpha_NT.py
=============================
Heatmap of AEI solutions (all checks: WKB + beta + shear) in the
(mdot, alpha_visc) parameter space for the Novikov-Thorne model.

Each cell (mdot_i, alpha_j) is coloured by the median physical radius [r_g]
of all valid AEI solutions found in that cell, aggregated over all spin
values in A_GRID and all radial points.

H/r and the WKB upper bound are taken directly from the NT model (hr_arr).

Usage:
    python aei_heatmap_mdot_alpha_NT.py
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
    r_ilr,
    solve_k_aei, compute_beta, compute_dQdr,
    ALPHA_VISC, mm, _make_interp,
)
from AEI_setups.nt_disc import disk_model_NT, nt_boundaries

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# NT scaling factor (eta = 0.1 convention, same as notebook)
MDOT_SCALE_NT = 10.0

# Parameter grids
N_MDOT  = 40
N_ALPHA = 30
MDOT_GRID  = np.logspace(-5, 0, N_MDOT)   # physical mdot
ALPHA_GRID = np.logspace(-3, 0, N_ALPHA)                    # alpha_visc

# Spin grid — aggregated over (same 19 points as notebook)
A_GRID = np.linspace(-0.99, 0.99, 19)

# Radial grid — global, shared across all spins
N_R          = 200
R_RISCO_MIN  = 1.0    # start at ISCO
R_MAX_PHYS   = 1000.0
R_RISCO_GRID = np.geomspace(R_RISCO_MIN, R_MAX_PHYS, N_R)  # r/r_ISCO

# WKB lower bound; upper bound is 1/hr(r) from the model
K_MIN_WKB = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# SCAN
# ══════════════════════════════════════════════════════════════════════════════

def build_heatmap():
    """
    For each (mdot_i, alpha_j) cell accumulate physical radii [r_g] of all
    valid AEI solutions over the full A_GRID.
    Returns r_map of shape (N_MDOT, N_ALPHA): median r [r_g] per cell.
    """
    r_accum = [[[] for _ in range(N_ALPHA)] for _ in range(N_MDOT)]
    hr_accum = [[[] for _ in range(N_ALPHA)] for _ in range(N_MDOT)]

    total = N_MDOT * N_ALPHA * len(A_GRID)
    count = 0

    for i, mdot in enumerate(MDOT_GRID):
        mdot = mdot * MDOT_SCALE_NT
        for j, alpha in enumerate(ALPHA_GRID):
            for a_val in A_GRID:
                count += 1
                if count % 1000 == 0:
                    print(f"  {count}/{total}  ({100*count/total:.0f}%)", end='\r')

                isco_val = float(r_isco(a_val))
                r_vec    = R_RISCO_GRID * isco_val   # physical radii [r_g]

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

                # WKB: 1 <= k < 1/hr(r)  — hr from NT model
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

                r_accum[i][j].extend(r_vec[mask].tolist())
                hr_accum[i][j].extend(hr_arr[mask].tolist())

    print()

    r_map = np.full((N_MDOT, N_ALPHA), np.nan)
    for i in range(N_MDOT):
        for j in range(N_ALPHA):
            if r_accum[i][j]:
                r_map[i, j] = np.median(r_accum[i][j])

    hr_map = np.full((N_MDOT, N_ALPHA), np.nan)
    for i in range(N_MDOT):
        for j in range(N_ALPHA):
            if hr_accum[i][j]:
                hr_map[i, j] = np.max(hr_accum[i][j])
    
    return r_map, hr_map


# ══════════════════════════════════════════════════════════════════════════════
# BOUNDARY MAPS  r_AB and r_BC as functions of (mdot, alpha)
# ══════════════════════════════════════════════════════════════════════════════

def compute_boundary_maps():
    """
    For each (mdot_i, alpha_j) compute the spin-averaged r_AB and r_BC [r_g].
    Returns two (N_MDOT, N_ALPHA) arrays.
    """
    print("  Computing r_AB and r_BC maps...", end=' ', flush=True)
    r_AB_map = np.full((N_MDOT, N_ALPHA), np.nan)
    r_BC_map = np.full((N_MDOT, N_ALPHA), np.nan)

    for i, mdot in enumerate(MDOT_GRID):
        mdot = mdot * MDOT_SCALE_NT
        for j, alpha in enumerate(ALPHA_GRID):
            ab_vals, bc_vals = [], []
            for a_val in A_GRID:
                try:
                    r_AB, r_BC, _ = nt_boundaries(a_val, mdot, alpha=alpha, M=M_BH)
                    ab_vals.append(r_AB)
                    bc_vals.append(r_BC)
                except Exception:
                    pass
            if ab_vals:
                r_AB_map[i, j] = np.nanmean(ab_vals)
            if bc_vals:
                r_BC_map[i, j] = np.nanmean(bc_vals)

    print("done")
    return r_AB_map, r_BC_map

# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(r_map, r_AB_map, r_BC_map, hr_map):
    valid_r = r_map[np.isfinite(r_map)]
    if valid_r.size == 0:
        print("WARNING: no valid solutions to plot.")
        return
    valid_hr = hr_map[np.isfinite(hr_map) & (hr_map > 0)]
    if valid_hr.size == 0:
        print("WARNING: no valid hr values to plot.")
        return

    norm1 = LogNorm(vmin=np.nanmin(valid_r[valid_r > 0]),
                    vmax=np.nanmax(valid_r[valid_r > 0]))
    norm2 = LogNorm(vmin=valid_hr.min(), vmax=valid_hr.max())

    fig, axs = plt.subplots(1, 2, figsize=(18, 7))

    cmap = plt.cm.plasma

    ax = axs[0]
    im = ax.pcolormesh(
        ALPHA_GRID, MDOT_GRID, r_map,
        norm=norm1, cmap=cmap, shading='auto',
    )
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'median $r$  [r$_g$]', fontsize=11)

    # ── ILR contour ────────────────────────────────────────────────────
    # r_ILR depends only on spin; average over A_GRID gives a single value.
    # The contour r_median(mdot, alpha) = r_ILR_mean marks where the median
    # AEI solution transitions from inside to outside the ILR cavity.
    ilr_vals = np.array([r_ilr(float(a)) for a in A_GRID])
    r_ilr_mean = float(np.nanmean(ilr_vals))
    print(f'   ILR mean over spins: {r_ilr_mean:.2f} rg')

    # fill NaN with large value so contour works on the full grid
    r_map_filled = np.where(np.isfinite(r_map), r_map, np.nanmax(r_map) * 10)
    if valid_r.min() < r_ilr_mean < valid_r.max():
        cs = ax.contour(ALPHA_GRID, MDOT_GRID, r_map_filled,
                        levels=[r_ilr_mean],
                        colors=['black'], linewidths=2.0, linestyles='-',
                        zorder=5)
        #ax.clabel(cs, fmt=rf'$r_{{\rm ILR}}$ mean = {r_ilr_mean:.1f} r$_g$',
        #          fontsize=8, colors='black')
    else:
        print(f'   WARNING: r_ILR_mean={r_ilr_mean:.1f} outside r_map range '
              f'[{valid_r.min():.1f}, {valid_r.max():.1f}] — contour not drawn')

    # ── r_AB and r_BC contours ────────────────────────────────────────
    # These depend on (mdot, alpha), so they produce real 2D curves.
    # We overlay them directly as contour lines on the heatmap.
    boundary_styles = [
        (r_AB_map, '#00e5ff', r'$r_{AB}$ mean'),   # cyan
        (r_BC_map, 'green', r'$r_{BC}$ mean'),  
    ]
    for bmap, bcolor, blabel in boundary_styles:
        bmap_filled = np.where(np.isfinite(bmap), bmap, np.nanmax(bmap[np.isfinite(bmap)]) * 10)
        b_valid = bmap[np.isfinite(bmap)]
        if b_valid.size == 0:
            continue
        # pick a representative contour level: median of the boundary map
        b_level = float(np.nanmedian(bmap))
        # draw the actual 2D boundary map as a single contour line
        try:
            cs_b = ax.contour(ALPHA_GRID, MDOT_GRID, bmap_filled,
                              levels=[b_level],
                              colors=[bcolor], linewidths=2.0,
                              linestyles='--', zorder=6)
            #ax.clabel(cs_b, fmt=f'{blabel} = {{:.1f}} r$_g$'.format(b_level),
            #          fontsize=8, colors=bcolor)
        except Exception:
            pass

    # fraction of valid cells

    n_valid = np.sum(np.isfinite(r_map))
    ax.text(0.97, 0.03,
            f'{n_valid}/{r_map.size} valid cells ({100*n_valid/r_map.size:.1f}%)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black',   lw=2.0, ls='-',  label=rf'$r_{{\rm ILR}}$ mean = {r_ilr_mean:.1f} r$_g$'),
        Line2D([0], [0], color='#00e5ff', lw=2.0, ls='--', label=r'$r_{AB}$ mean'),
        Line2D([0], [0], color='green', lw=2.0, ls='--', label=r'$r_{BC}$ mean'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left',
            facecolor='black', labelcolor='white', framealpha=0.6)

    """
    ax.set_title(
        r"AEI solutions — Novikov-Thorne" "\n"
        r"All checks (WKB + $\beta$ + shear), $H/r$ from model" "\n"
        rf"colour = median $r$ [r$_g$],  aggregated over {len(A_GRID)} spin values",
        fontsize=11
    )
    """

    ##### H/r map
    ax = axs[1]
    im2 = ax.pcolormesh(
        ALPHA_GRID, MDOT_GRID, hr_map,
        norm=norm2, cmap=cmap, shading='auto',
    )
    cb2 = plt.colorbar(im2, ax=ax, pad=0.02)
    cb2.set_label(r'maximum $H/r$', fontsize=11)

    # H/r limits
    cs_b2 = ax.contour(ALPHA_GRID, MDOT_GRID, hr_map,
                        levels=[0.1, 0.3],
                        colors=["cyan", "green"], linewidths=2.0,
                        linestyles='--', zorder=6)
    legend_elements = [
        Line2D([0], [0], color='cyan', lw=2.0, ls='--', label=r'thin disc limit ($H/r = 0.1$)'),
        Line2D([0], [0], color='green', lw=2.0, ls='--', label=r'slim disc ($H/r = 0.3$)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left',
            facecolor='black', labelcolor='white', framealpha=0.6)

    n_valid = np.sum(np.isfinite(hr_map))
    ax.text(0.97, 0.03,
        f'{n_valid}/{hr_map.size} valid cells ({100*n_valid/hr_map.size:.1f}%)',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8, color='white',
        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))

    for ax in axs:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\alpha_{\rm visc}$', fontsize=13)
        ax.set_ylabel(r'$\dot{m} = \dot{M}/\dot{M}_{\rm Edd}$', fontsize=13)
        ax.grid(True, which='both', alpha=0.15, lw=0.5)

    plt.tight_layout()
    plt.savefig('aei_heatmap_mdot_alpha_NT.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('aei_heatmap_mdot_alpha_NT.png', bbox_inches='tight', dpi=150)
    print("   -> saved aei_heatmap_mdot_alpha_NT.pdf / .png")
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  AEI Heatmap: mdot vs alpha_visc — Novikov-Thorne")
    print(f"  mdot  : {N_MDOT} values in [{MDOT_GRID[0]:.2e}, {MDOT_GRID[-1]:.2e}]")
    print(f"          (x{MDOT_SCALE_NT} NT scaling, eta=0.1)")
    print(f"  alpha : {N_ALPHA} values in [{ALPHA_GRID[0]:.2e}, {ALPHA_GRID[-1]:.2e}]")
    print(f"  spin  : {len(A_GRID)} values in [{A_GRID[0]:.1f}, {A_GRID[-1]:.1f}] (aggregated)")
    print(f"  r grid: {N_R} points, R_MAX = {R_MAX_PHYS} rg")
    print("=" * 65)

    print("\n> Building heatmap...")
    r_map, hr_map = build_heatmap()
    n = np.sum(np.isfinite(r_map))
    print(f"   -> {n} filled cells out of {r_map.size} ({100*n/r_map.size:.1f}%)")

    r_AB_map, r_BC_map = compute_boundary_maps()

    print("\n> Plotting...")
    plot_heatmap(r_map, r_AB_map, r_BC_map, hr_map)
