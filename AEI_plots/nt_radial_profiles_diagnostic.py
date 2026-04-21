"""
nt_radial_profiles_diagnostic.py  — notebook-ready version
===========================================================
Drop this file in the project root (or AEI_plots/) and call:

    from nt_radial_profiles_diagnostic import plot_nt_profiles

    fig = plot_nt_profiles(a=0.9, mdot=0.02, alpha=0.1)

or loop over parameters:

    for a in [0.0, 0.5, 0.9]:
        plot_nt_profiles(a=a, mdot=0.02, alpha=0.1,
                         save=True, outdir='figures')
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')

from setup import r_isco, M_BH, nu_phi, Rg_SUN
from aei_2.aei_setup import r_ilr, r_olr, compute_beta
from aei_2.nt_disc import disk_model_NT

# ── stile zone ────────────────────────────────────────────────────────────────
ZONE_COLOR = {'A': '#3b82f6', 'B': '#f97316', 'C': '#22c55e'}


# ═══════════════════════════════════════════════════════════════════════════════
# helper
# ═══════════════════════════════════════════════════════════════════════════════

def _find_Q_extrema(r, Sigma, B0, a, M, dr_frac=0.01):
    """Trova r del max e min locale di Q_AEI = Ω·Σ/B² tramite dQ/dr."""
    Omega = 2.0 * np.pi * nu_phi(r, a, M)
    Q     = Omega * Sigma / np.maximum(B0**2, 1e-300)
    dr    = dr_frac * r
    rp    = r + dr
    log_r = np.log(r)
    log_Q = np.log(np.maximum(Q, 1e-300))
    Q_rp  = np.exp(np.interp(np.log(rp), log_r, log_Q))
    dQdr  = (Q_rp - Q) / dr
    sc    = np.where(np.diff(np.sign(dQdr)))[0]
    i_max = next((i for i in sc if dQdr[i] > 0), None)
    i_min = next((i for i in sc if dQdr[i] < 0), None)
    return (Q,
            float(r[i_max]) if i_max is not None else None,
            float(r[i_min]) if i_min is not None else None)


# ═══════════════════════════════════════════════════════════════════════════════
# funzione principale
# ═══════════════════════════════════════════════════════════════════════════════

def plot_nt_profiles(
    a      = 0.9,
    mdot   = 0.02,
    alpha  = 0.1,
    M      = M_BH,
    n_r    = 600,
    r_min_fac = 1.001,
    r_max_fac = 250.0,
    show_Q_markers = False,   # mostra r(Q_max) e r(Q_min) — off di default
    save   = False,
    outdir = '.',
    outfile = None,           # None → nome automatico
    figsize = (7, 8),
):
    """
    4 pannelli sovrapposti (asse x condiviso): B₀, Σ, H/r, β
    per il modello NT a parametri fissati.

    Parameters
    ----------
    a, mdot, alpha, M : parametri fisici del disco
    n_r               : punti radiali
    r_min_fac/r_max_fac : range radiale in unità di r_ISCO
    show_Q_markers    : bool — aggiunge linee verticali per r(Q_max), r(Q_min)
    save              : bool — salva PDF e PNG
    outdir            : str  — directory output
    outfile           : str|None — nome base file; None → generato da parametri
    figsize           : tuple

    Returns
    -------
    fig : matplotlib.Figure
    info : dict  — raggi chiave calcolati
    """
    # ── griglia radiale ───────────────────────────────────────────────────
    rISCO = float(r_isco(a))
    r_vec = np.geomspace(rISCO * r_min_fac, rISCO * r_max_fac, n_r)

    # ── profili fisici ────────────────────────────────────────────────────
    B0_arr, Sig_arr, cs_arr, hr_arr, zone_arr, info_d = disk_model_NT(
        r_vec, a, mdot, alpha_visc=alpha, hr=None, M=M
    )
    r_AB = info_d['r_AB']
    r_BC = info_d['r_BC']

    beta_arr = compute_beta(B0_arr, Sig_arr, cs_arr, r_vec, hr_arr, M)

    # ── risonanze ─────────────────────────────────────────────────────────
    r_ILR = r_ilr(a, M=M)
    r_OLR = r_olr(a, M=M)

    # ── Q_AEI ─────────────────────────────────────────────────────────────
    _, r_max_Q, r_min_Q = _find_Q_extrema(r_vec, Sig_arr, B0_arr, a, M)

    # ── marcatori verticali ───────────────────────────────────────────────
    markers = [(rISCO, 'black', '-', r'$r_{\rm ISCO}$', 2.0)]
    if np.isfinite(r_ILR):
        markers.append((r_ILR, '#06b6d4', '--', r'$r_{\rm ILR}$', 1.8))
    if np.isfinite(r_OLR):
        markers.append((r_OLR, '#d946ef', '-.', r'$r_{\rm OLR}$', 1.8))
    if r_AB > rISCO * 1.01:
        markers.append((r_AB, '#f97316', '--', r'$r_{AB}$', 1.5))
    if r_BC > r_AB * 1.01:
        markers.append((r_BC, '#22c55e', '--', r'$r_{BC}$', 1.5))
    if show_Q_markers:
        if r_max_Q is not None:
            markers.append((r_max_Q, 'gold',   '--', r'$Q_{\rm AEI}$ max', 1.3))
        if r_min_Q is not None:
            markers.append((r_min_Q, 'tomato', '--', r'$Q_{\rm AEI}$ min', 1.3))

    # ── panels: (data, ylabel, log_y, hlines) ────────────────────────────
    panels = [
        (B0_arr,   r'$B_0$  [G]',              True,  []),
        (Sig_arr,  r'$\Sigma$  [g cm$^{-2}$]', True,  []),
        (hr_arr,   r'$H/r$',                   True,
            [(0.1, 'steelblue', ':', r'$H/r=0.1$'),
             (0.3, 'crimson',   ':', r'$H/r=0.3$')]),
        (beta_arr, r'$\beta$',                 True,
            [(1.0, 'crimson', '--', r'$\beta=1$')]),
    ]

    # ── figura ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        4, 1, figsize=figsize, sharex=True,
        gridspec_kw=dict(hspace=0.06),
    )

    def _shade(ax):
        for zname, zlo, zhi in [('A', rISCO, r_AB),
                                 ('B', r_AB,  r_BC),
                                 ('C', r_BC,  r_vec[-1])]:
            if zhi > zlo:
                ax.axvspan(zlo, zhi, alpha=0.06,
                           color=ZONE_COLOR.get(zname, 'gray'))

    def _vlines(ax):
        for rv, col, ls, lbl, lw in markers:
            ax.axvline(rv, color=col, ls=ls, lw=lw, alpha=0.85)

    for k, (ax, (data, ylabel, log_y, hlines)) in enumerate(zip(axes, panels)):
        _shade(ax)
        _vlines(ax)

        for zname, col in ZONE_COLOR.items():
            mask = zone_arr == zname
            ok   = mask & (data > 0) if log_y else mask
            if ok.sum() >= 2:
                ax.plot(r_vec[ok], data[ok], color=col, lw=2.0, zorder=3)

        for yval, hcol, hls, hlbl in hlines:
            ax.axhline(yval, color=hcol, ls=hls, lw=1.4, alpha=0.85,
                       label=hlbl)

        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.15, which='both')
        ax.set_xlim(r_vec[0], r_vec[-1])
        if k < 3:
            ax.tick_params(axis='x', labelbottom=False)

    axes[-1].set_xlabel(r'$r$  [$r_g$]', fontsize=11)

    # ── legenda (pannello superiore) ──────────────────────────────────────
    legend_els = [
        Line2D([0], [0], color=col, ls=ls, lw=lw, label=lbl)
        for rv, col, ls, lbl, lw in markers
    ] + [
        Patch(facecolor=col, alpha=0.35, label=f'Zone {z}')
        for z, col in ZONE_COLOR.items()
    ]
    axes[0].legend(handles=legend_els, fontsize=7.5, ncol=2,
                   loc='upper right', framealpha=0.88,
                   borderpad=0.5, labelspacing=0.25, handlelength=1.6)

    plt.tight_layout()

    # ── salvataggio opzionale ─────────────────────────────────────────────
    if save:
        import os
        os.makedirs(outdir, exist_ok=True)
        base = outfile or f'nt_profiles_a{a:.2f}_mdot{mdot:.3g}_alpha{alpha}'
        base = os.path.join(outdir, base)
        for ext in ('pdf', 'png'):
            fig.savefig(f'{base}.{ext}', bbox_inches='tight', dpi=150)
            print(f'  → {base}.{ext}')

    # ── summary numerico ──────────────────────────────────────────────────
    out_info = dict(rISCO=rISCO, r_AB=r_AB, r_BC=r_BC,
                    r_ILR=r_ILR, r_OLR=r_OLR,
                    r_max_Q=r_max_Q, r_min_Q=r_min_Q)

    return fig, out_info


# ═══════════════════════════════════════════════════════════════════════════════
# script standalone (invariato)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    fig, info = plot_nt_profiles(
        a=0.9, mdot=0.02, alpha=0.1,
        save=True, outdir='.',
    )
    print('\nKey radii:')
    for k, v in info.items():
        if v is not None:
            print(f'  {k:10s} = {v:.3f} rg')
    plt.show()