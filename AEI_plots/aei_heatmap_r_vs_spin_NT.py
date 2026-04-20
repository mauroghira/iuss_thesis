"""
aei_heatmap_r_vs_spin_NT.py
============================
Heatmap of AEI solutions (all checks: WKB + beta + shear) in the
(spin a, r/r_ISCO) plane for the Novikov-Thorne disk model,
coloured by the median dimensionless wavenumber k.

The free parameter here is mdot (not B00/Sigma0 as in the Simple models).
Grid: A_GRID x MDOT_GRID  (same philosophy — one column per spin,
one row per r/r_ISCO point, median k over all mdot values per cell).

H/r is fixed (HOR = 0.05 as default); the WKB range is 1 <= k < 1/HOR.

Overlaid curves:
  - ISCO : r/r_ISCO = 1
  - ILR  : r_ILR(a) / r_ISCO(a)
  - OLR  : r_OLR(a) / r_ISCO(a)
  - CR   : r_CR(a)  / r_ISCO(a)

Usage:
    python aei_heatmap_r_vs_spin_NT.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')

from setup import M_BH, NU0, r_isco, nu_phi
from AEI_setups.aei_common import (
    solve_k_aei, compute_beta, compute_dQdr,
    r_ilr, r_olr, r_corotation,
    HOR, ALPHA_VISC, mm, _make_interp,
)
from aei_2.nt_disc import disk_model_NT

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# H/r for the WKB check and sound speed
HOR_PLOT  = HOR          # 0.05 — change here if needed
K_MIN_WKB = 1.0
# K_MAX_WKB is now per-point: k < r/H = 1/hr(r), computed from hr_arr returned by the model

# NT scaling factor (eta = 0.1 convention)
MDOT_SCALE_NT = 10.0

# Spin grid — one column per value
A_GRID = np.linspace(-0.99, 0.99, 60)

# mdot grid: same base range as notebook (_MDOT_BASE = logspace(-5,0,12)),
# scaled by MDOT_SCALE_NT; we use a denser grid here
N_MDOT   = 40
MDOT_GRID = np.logspace(-5, 0, N_MDOT) * MDOT_SCALE_NT

# Radial grid
N_R              = 300
R_RISCO_MIN      = 0.8
R_MAX_PHYS       = 1000.0
R_RISCO_MAX_GRID = R_MAX_PHYS
R_RISCO_GRID     = np.geomspace(R_RISCO_MIN, R_RISCO_MAX_GRID, N_R)

# Upper y-limit set automatically from data (highest r/r_ISCO with valid solutions)

# Fine spin grid for smooth resonance curves
A_CURVE = np.linspace(-0.99, 0.99, 500)

# ══════════════════════════════════════════════════════════════════════════════
# SCAN
# ══════════════════════════════════════════════════════════════════════════════

def build_heatmap():
    """
    For each (a_i, r_j/r_ISCO) cell accumulate k values over all mdot
    values that pass all checks. Return median-k map (N_A x N_R).
    """
    N_A = len(A_GRID)
    k_accum = [[[] for _ in range(N_R)] for _ in range(N_A)]

    total = N_A * N_MDOT
    count = 0

    # Global physical mask: only rows >= ISCO (r/r_ISCO >= 1).
    # The upper limit is NOT cut per-spin: we use the full shared grid
    # and let the data determine where solutions actually exist.
    global_phys_mask    = R_RISCO_GRID >= 1.0
    global_phys_indices = np.where(global_phys_mask)[0]

    for i, a_val in enumerate(A_GRID):
        isco_val     = float(r_isco(a_val))
        r_vec_phys   = R_RISCO_GRID[global_phys_mask] * isco_val
        phys_indices = global_phys_indices

        for mdot in MDOT_GRID:
            count += 1
            if count % 200 == 0:
                print(f"  {count}/{total}  ({100*count/total:.0f}%)", end='\r')

            try:
                result = disk_model_NT(r_vec_phys, a_val, mdot,
                                       alpha_visc=ALPHA_VISC,
                                       hr=HOR_PLOT, M=M_BH)
            except Exception:
                continue

            B0_arr, Sigma_arr, cs_arr, hr_arr, _, _ = result

            k_arr = solve_k_aei(r_vec_phys, a_val, B0_arr, Sigma_arr, cs_arr,
                                m=mm, M=M_BH)

            mask = np.isfinite(k_arr) & (k_arr > 0)
            if not np.any(mask): continue

            # WKB upper bound: k < r/H = 1/hr(r)  — per-point, from the NT model
            mask &= (k_arr >= K_MIN_WKB) & (k_arr < 1.0 / hr_arr)
            if not np.any(mask): continue

            # beta uses the physical H from the NT model (hr_arr), not a fixed HOR
            beta_arr = compute_beta(B0_arr, Sigma_arr, cs_arr,
                                    r_vec_phys, hr_arr, M_BH)
            mask &= (beta_arr <= 1.0)
            if not np.any(mask): continue

            _B0i  = _make_interp(r_vec_phys, B0_arr)
            _Sigi = _make_interp(r_vec_phys, Sigma_arr)
            dQdr  = compute_dQdr(r_vec_phys, a_val, _B0i, _Sigi, M_BH)
            mask &= (dQdr > 0)
            if not np.any(mask): continue

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
    valid_k = k_map[np.isfinite(k_map)]
    if valid_k.size == 0:
        print("WARNING: no valid solutions to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    norm = LogNorm(vmin=valid_k.min(), vmax=valid_k.max())
    cmap = plt.cm.viridis

    im = ax.pcolormesh(
        A_GRID, R_RISCO_GRID, k_map.T,
        norm=norm, cmap=cmap, shading='auto',
    )
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r'median $k$  (dimensionless)', fontsize=11)

    ax.set_yscale('log')
    ax.axhline(1.0, color='white', lw=2.0, ls='-',
               label=r'ISCO  ($r/r_{\rm ISCO}=1$)', zorder=5)

    res_style = {
        'ILR': dict(color='dodgerblue', lw=1.8, ls='--',
                    label=r'$r_{\rm ILR}/r_{\rm ISCO}$'),
        'OLR': dict(color='#ff00ff',    lw=2.2, ls='--',
                    label=r'$r_{\rm OLR}/r_{\rm ISCO}$'),
        'CR':  dict(color='gold',       lw=1.8, ls=':',
                    label=r'$r_{\rm CR}/r_{\rm ISCO}$'),
    }
    for name, arr in curves.items():
        mask = np.isfinite(arr)
        if mask.any():
            ax.plot(A_CURVE[mask], arr[mask], zorder=6, **res_style[name])

    ax.set_xlabel(r'Spin  $a$', fontsize=12)
    ax.set_ylabel(r'$r \,/\, r_{\rm ISCO}$', fontsize=12)
    ax.set_xlim(A_GRID[0] - 0.02, A_GRID[-1] + 0.02)
    # upper y-limit: highest r/r_ISCO row with any valid solution + 20% margin
    valid_rows = np.where(np.any(np.isfinite(k_map), axis=0))[0]
    r_max_data = R_RISCO_GRID[valid_rows[-1]] * 1.2 if valid_rows.size else R_RISCO_MAX_GRID
    ax.set_ylim(R_RISCO_MIN, r_max_data)
    ax.grid(True, which='both', alpha=0.15, lw=0.5)
    ax.legend(fontsize=9, loc='upper left')
    """
    ax.set_title(
        rf"AEI solutions — Novikov-Thorne  ($\alpha_{{\rm visc}}={ALPHA_VISC}$,"
        r" $H/r$ from NT model)"
        "\n"
        r"All checks (WKB + $\beta$ + shear) — colour = median $k$",
        fontsize=12
    )
    """

    plt.tight_layout()
    plt.savefig('aei_heatmap_r_vs_spin_NT.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('aei_heatmap_r_vs_spin_NT.png', bbox_inches='tight', dpi=150)
    print("   -> saved aei_heatmap_r_vs_spin_NT.pdf / .png")
    plt.show()


def diagnosis():
    a_test   = 0
    mdot_test = MDOT_GRID[len(MDOT_GRID)//2]   # mdot mediano
    isco_test = float(r_isco(a_test))

    r_fine = np.geomspace(isco_test * 1.01, isco_test * 4.0, 500)
    result = disk_model_NT(r_fine, a_test, mdot_test,
                        alpha_visc=ALPHA_VISC, hr=HOR_PLOT, M=M_BH)
    B0_t, Sig_t, cs_t, hr_t, _, _ = result

    k_t    = solve_k_aei(r_fine, a_test, B0_t, Sig_t, cs_t, m=mm, M=M_BH)
    beta_t = compute_beta(B0_t, Sig_t, cs_t, r_fine, hr_t, M_BH)
    dQ_t   = compute_dQdr(r_fine, a_test,
                        _make_interp(r_fine, B0_t),
                        _make_interp(r_fine, Sig_t), M_BH)

    m_k    = np.isfinite(k_t) & (k_t >= K_MIN_WKB) & (k_t < 1.0/hr_t)
    m_beta = beta_t <= 1.0
    m_dq   = dQ_t > 0

    r_norm = r_fine / isco_test
    print(f"\nDiagnostica a a={a_test}, mdot={mdot_test:.2e}")
    print(f"  Primo punto con k ok    : r/rISCO = {r_norm[m_k][0]:.3f}" if m_k.any() else "  k: nessun punto valido")
    print(f"  Primo punto con beta ok : r/rISCO = {r_norm[m_beta][0]:.3f}" if m_beta.any() else "  beta: nessun punto valido")
    print(f"  Primo punto con dQdr ok : r/rISCO = {r_norm[m_dq][0]:.3f}" if m_dq.any() else "  dQdr: nessun punto valido")
    print(f"  Primo punto con tutti ok: r/rISCO = {r_norm[m_k & m_beta & m_dq][0]:.3f}" if (m_k & m_beta & m_dq).any() else "  nessuna soluzione")

    # calcola Q(r) esplicitamente
    Omega_t = 2 * np.pi * nu_phi(r_fine, a_test, M_BH)  # importa nu_phi da setup
    Q_func  = Omega_t * Sig_t / B0_t**2

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(r_norm, Q_func)
    axes[0].set_ylabel(r'$Q = \Omega_\phi \Sigma / B_0^2$')
    #axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(r_norm, dQ_t)
    axes[1].axhline(0, color='red', ls='--')
    axes[1].set_ylabel(r'$dQ/dr$')
    axes[1].set_xlabel(r'$r / r_{\rm ISCO}$')
    axes[1].set_ylim(-np.abs(dQ_t).max()*0.1,
                    np.abs(dQ_t[r_fine < isco_test*3]).max()*1.2)
    axes[1].grid(True, alpha=0.2)

    sign_changes = np.where(np.diff(np.sign(dQ_t)))[0]

    # massimo locale: primo cambio + → -
    i_max_local = next((i for i in sign_changes if dQ_t[i] > 0), None)
    # minimo locale: primo cambio - → + (dopo il massimo)
    i_min_local = next((i for i in sign_changes if dQ_t[i] < 0), None)

    if i_max_local is not None:
        r_max_Q = r_norm[i_max_local]
        axes[0].axvline(r_max_Q, color='orange', ls='--',
                        label=f'Q max locale @ {r_max_Q:.2f} rISCO')
        axes[1].axvline(r_max_Q, color='orange', ls='--')

    if i_min_local is not None:
        r_min_Q = r_norm[i_min_local]
        axes[0].axvline(r_min_Q, color='red', ls='--',
                        label=f'Q min locale (r_match) @ {r_min_Q:.2f} rISCO')
        axes[1].axvline(r_min_Q, color='red', ls='--')

    #axes[0].set_title(f'NT model — a={a_test}, mdot={mdot_test:.2e}\n'
    #                f'Q(r) e dQ/dr nella zona interna')
    
    plt.tight_layout()
    plt.savefig('diagnostic.png', bbox_inches='tight', dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  AEI Heatmap: r/r_ISCO vs spin — Novikov-Thorne")
    print(f"  Spin grid  : {len(A_GRID)} values in [{A_GRID[0]:.1f}, {A_GRID[-1]:.1f}]")
    print(f"  mdot grid  : {N_MDOT} values in [{MDOT_GRID[0]:.2e}, {MDOT_GRID[-1]:.2e}]")
    print(f"               (x{MDOT_SCALE_NT} NT scaling, eta=0.1)")
    print(f"  H/r        : from NT model (per-point)  |  WKB: k >= {K_MIN_WKB:.0f}, k < 1/hr(r)")
    print(f"  alpha_visc : {ALPHA_VISC}")
    print(f"  R_MAX_PHYS : {R_MAX_PHYS} rg  |  y-limit: data-driven")
    print("=" * 65)

    # ── diagnostica: quale check taglia il bordo inferiore? ──────────────────────
    diagnosis()

    print("\n> Building heatmap...")
    k_map = build_heatmap()
    n = np.sum(np.isfinite(k_map))
    print(f"   -> {n} filled cells out of {k_map.size} ({100*n/k_map.size:.1f}%)")

    curves = compute_resonance_curves()

    print("\n> Plotting...")
    plot_heatmap(k_map, curves)

