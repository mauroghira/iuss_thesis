import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
from setup import *
from plts_funcs import *

set_style()
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

stiles = ['-', '--', '-.', ':', ""]
colors = ["blue", "orange", "lightgreen", "red", "purple"]

def find_continuous_segments(mask):
    segs = []; in_seg = False
    for i, v in enumerate(mask):
        if v and not in_seg:  start = i; in_seg = True
        elif not v and in_seg: segs.append((start, i-1)); in_seg = False
    if in_seg: segs.append((start, len(mask)-1))
    return segs

def plots_frq(ax, r_vals, all_frequencies, mm, nn):
    """Plot boundary curves (no labels – handled by shared legend)."""
    # mΩ + κ (red solid)
    ax.plot(r_vals, all_frequencies[1], color="#C0392B",   linestyle="-",  linewidth=1)
    
    if nn > 0:
        # mΩ + √n Ω⊥ 
        ax.plot(r_vals, all_frequencies[3], color="#D4AC0D", linestyle="-.", linewidth=1)
    
    if mm > 0:
        #ax.plot(r_vals, all_frequencies[0], color="black", linestyle=":",  linewidth=1)  # mΩ
        ax.plot(r_vals, all_frequencies[2], color="#C0392B",   linestyle="-",  linewidth=1)  # mΩ - κ
        if nn>0:
            ax.plot(r_vals, all_frequencies[4], color="#D4AC0D", linestyle="-.", linewidth=1)  # mΩ - √n Ω⊥


######################################

def plot_target_wavy_trapped(ax, r_vals, freq_plus_kappa, freq_minus_kappa,
                              freq_plus_n20, freq_minus_n20, target_freq, mm, nn,
                              legend_handles):
    """
    Plot target frequency as a wavy line only in physically allowed regions.
    p-modes: only the innermost continuous segment is kept.
    legend_handles (dict) is updated in-place for the shared figure legend.
    """
    n_waves = 20
    wave_amplitude = target_freq * 0.1
    log_r = np.log10(r_vals)
    log_r_norm = (log_r - log_r.min()) / (log_r.max() - log_r.min())
    wave = wave_amplitude * np.sin(2 * np.pi * n_waves * log_r_norm)
    target_wavy_pos = target_freq + wave
    target_wavy_neg = -target_freq + wave

    def _plot_first_segment(mask, y_arr, color, key, label):
        """Plot only the first (innermost) continuous segment."""
        segs = find_continuous_segments(mask)
        if segs:
            s, e = segs[0]
            if r_vals[e] < np.max(RIVR):
                ax.plot(r_vals[s:e+1], y_arr[s:e+1], color=color, linewidth=1, linestyle='-')
            if key not in legend_handles:
                legend_handles[key] = mlines.Line2D([], [], color=color, linewidth=1, label=label)

    def _plot_all_segments(mask, y_arr, color, key, label):
        """Plot all continuous segments (used for trapped modes)."""
        segs = find_continuous_segments(mask)
        for s, e in segs:
            if r_vals[e] < np.max(RIVR):
                ax.plot(r_vals[s:e+1], y_arr[s:e+1], color=color, linewidth=1, linestyle='-')
        if segs and key not in legend_handles:
            legend_handles[key] = mlines.Line2D([], [], color=color, linewidth=1, label=label)

    # ── n=0, any m: only p-modes ────────────────────────────────────────
    if nn == 0:
        p_mask = (target_wavy_pos >= freq_plus_kappa) & ~np.isnan(freq_plus_kappa)
        _plot_first_segment(p_mask, target_wavy_pos, 'darkorange', 'p', 'p-mode')
        if mm > 0:
            p_mask_neg = (target_wavy_pos <= freq_minus_kappa) & ~np.isnan(freq_minus_kappa)
            _plot_first_segment(p_mask_neg, target_wavy_pos, 'darkorange', 'p', 'p-mode')

    # ── m=0, n>0: g-modes + p-modes ─────────────────────────────────────
    elif mm == 0 and nn > 0:
        g_mask = (target_wavy_pos <= freq_plus_kappa) & (target_wavy_pos > 0) & ~np.isnan(freq_plus_kappa)
        _plot_all_segments(g_mask, target_wavy_pos, 'steelblue', 'g', 'g-mode')
        p_mask = (target_wavy_pos >= freq_plus_n20) & ~np.isnan(freq_plus_n20)
        _plot_first_segment(p_mask, target_wavy_pos, 'darkorange', 'p', 'p-mode')

    # ── m>0, n>0: g-modes + c-modes + p-modes ───────────────────────────
    elif mm > 0 and nn > 0:
        g_mask = ((target_wavy_pos >= freq_minus_kappa) & (target_wavy_pos <= freq_plus_kappa)
                  & ~np.isnan(freq_minus_kappa) & ~np.isnan(freq_plus_kappa))
        _plot_all_segments(g_mask, target_wavy_pos, 'steelblue', 'g', 'g-mode')

        positive_n20 = (freq_minus_n20 > 0) & ~np.isnan(freq_minus_n20)
        negative_n20 = (freq_minus_n20 < 0) & ~np.isnan(freq_minus_n20)

        c_pos = (target_wavy_pos <= freq_minus_n20) & positive_n20 & (target_wavy_pos > 0)
        _plot_all_segments(c_pos, target_wavy_pos, 'brown', 'c', 'c-mode')
        c_neg = (-target_freq <= freq_minus_n20) & negative_n20
        _plot_first_segment(c_neg, target_wavy_neg, 'brown', 'c', 'c-mode')

        p_upper = (target_wavy_pos >= freq_plus_n20) & ~np.isnan(freq_plus_n20)
        _plot_first_segment(p_upper, target_wavy_pos, 'darkorange', 'p', 'p-mode')

        p_lower_neg = (target_wavy_neg <= freq_minus_n20) & negative_n20
        _plot_first_segment(p_lower_neg, target_wavy_neg, 'darkorange', 'p', 'p-mode')

    # ── m>0, n=0: only p-modes ───────────────────────────────────────────
    elif mm > 0 and nn == 0:
        p_mask = (target_wavy_pos >= freq_plus_kappa) & ~np.isnan(freq_plus_kappa)
        _plot_first_segment(p_mask, target_wavy_pos, 'darkorange', 'p', 'p-mode')

#############################################

params = {
    "m": (0, 3, 4),
    "j": (0, 3, 4),
    "a": (-1, 1, 201),
    "rivr": (1, 200, 200)
}
labels = list(params.keys())
param_vectors, mesh_arrays = create_param_grid(params)

m, j, A, RIVR = mesh_arrays

RIVR = np.maximum(RIVR, r_isco(A))

# ── frequencies on the grid (computed once outside the plot loop) ────
Omega      = nu_phi(RIVR, A)    # azimuthal orbital frequency
Omega_perp = nu_theta(RIVR, A)  # vertical epicyclic frequency
kappa      = nu_r(RIVR, A)      # radial epicyclic frequency

# ── choose spin slice ────────────────────────────────────────────────
ia = 200   # index into spin axis  (ia=0 → a ≈ -0.99)
a_val      = A[0, 0, ia, 0]
r_isco_val = r_isco(a_val)

NM, NJ = 2, 3   # m = 0..3,  n = 0..2

fig, axes = plt.subplots(NM, NJ, figsize=(7, 4.5), sharex=True)

legend_handles = {}   # filled in-place by plot_target_wavy_trapped

for im in range(NM):
    for ij in range(NJ):
        ax = axes[im, ij]
        fix_spines(ax)

        mm = int(round(m[im, ij, ia, 0]))
        nn = int(round(j[im, ij, ia, 0]))   # renamed j → n

        # boundary curves
        m_Omega         = mm * Omega[im, ij, ia, :]
        freq_plus_kappa = m_Omega + kappa[im, ij, ia, :]
        freq_plus_n20   = m_Omega + Omega_perp[im, ij, ia, :] * np.sqrt(nn)
        if mm > 0:
            freq_minus_kappa = m_Omega - kappa[im, ij, ia, :]
            freq_minus_n20   = m_Omega - Omega_perp[im, ij, ia, :] * np.sqrt(nn)
        else:
            freq_minus_kappa = freq_plus_kappa
            freq_minus_n20   = freq_plus_n20

        all_frequencies = [m_Omega, freq_plus_kappa, freq_minus_kappa,
                           freq_plus_n20, freq_minus_n20]
        has_negative = any(np.any(f < 0) for f in all_frequencies)

        r_vals = RIVR[im, ij, ia, :]

        # ── draw boundary curves ─────────────────────────────────────
        plots_frq(ax, r_vals, all_frequencies, mm, nn)

        # ── target band + wavy line (trapped modes only) ─────────────
        #ax.fill_between(r_vals, TARGET_MIN, TARGET_MAX, color='gray', alpha=0.2)
        plot_target_wavy_trapped(ax, r_vals, freq_plus_kappa, freq_minus_kappa,
                                 freq_plus_n20, freq_minus_n20, NU0, mm, nn,
                                 legend_handles)

        # ── ISCO vertical line ───────────────────────────────────────
        ax.axvline(r_isco_val, color='purple', linestyle='--', linewidth=0.8)

        # ── scale ────────────────────────────────────────────────────
        #ax.set_ylim(bottom=1e-5)
        #ax.set_yscale("symlog", linthresh=1e-5) if has_negative else ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(right=np.max(RIVR))

        # ── internal title ───────────────────────────────────────────
        ax.text(0.94, 0.94, fr"$m = {mm},  n = {nn}$",
                transform=ax.transAxes, #fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))

        # ── axis labels only on borders ──────────────────────────────
        if im == NM - 1:
            ax.set_xlabel(r"$R\,[R_g]$")
        else:
            ax.tick_params(labelbottom=False)

        if ij == 0:
            ax.set_ylabel("Frequency [Hz]")
        else:
            ax.tick_params(labelleft=False)

for im in range(NM):
    row_axes = [axes[im, ij] for ij in range(NJ)]
    row_tops = [ax.get_ylim()[1] for ax in row_axes]
    global_top = max(row_tops)
    for ax in row_axes:
        ax.set_ylim(bottom=1e-6, top=global_top*2)
        ax.set_yscale("log")

# ── single shared legend ─────────────────────────────────────────────
h_red   = mlines.Line2D([], [], color='#C0392B', lw=1, ls='-',
                         label=r'$m\nu_\varphi \pm \nu_r$')
h_green = mlines.Line2D([], [], color='#D4AC0D', lw=1, ls='-.',
                         label=r'$m\nu_\varphi \pm \sqrt{n}\,\nu_\vartheta$')
h_black = mlines.Line2D([], [], color='black', lw=1, ls=':',
                         label=r'$m\nu_\varphi$')
#h_gray  = mpatches.Patch(facecolor='gray', alpha=0.3, label='Target range')
h_isco  = mlines.Line2D([], [], color='purple', lw=0.8, ls='--', alpha=0.5,
                         label='ISCO')

static  = [h_red, h_green, h_isco]
dynamic = list(legend_handles.values())   # g-mode, c-mode, p-mode (appear as needed)

fig.legend(
    handles=static + dynamic,
    loc='lower center',
    ncol=len(static + dynamic),
    frameon=True,
    bbox_to_anchor=(0.5, -0.01)
)

plt.tight_layout(rect=[0, 0.04, 1, 0.995])
plt.savefig('cmodes_plots/regioni.pdf', bbox_inches='tight')
plt.show()

#################################################
#
###################################################

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── grid (higher resolution in a for smoother scatter/heatmap) ─────────
params = {
    "m":    (0, 2, 3),
    "j":    (0, 2, 3),       # n = 0, 1, 2
    "a":    (-1, 1, 120),
    "rivr": (1, 500, 400)
}
pv, ma = create_param_grid(params)
m_g, j_g, A, RIVR = ma
RIVR = np.maximum(RIVR, r_isco(A))

Om  = nu_phi(RIVR, A)       # Ω_φ
Op  = nu_theta(RIVR, A)     # Ω_θ
Kp  = nu_r(RIVR, A)         # κ

NM, NJ, NA, NR = m_g.shape
a_vec = A[0, 0, :, 0]       # 1-D spin vector
r_vec = RIVR[0, 0, 0, :]    # 1-D radius vector (at a fixed a; used only for ref)

# colour / marker coding for mode types
MODE_STYLE = {
    'g': dict(color='steelblue',    marker='o', label='g-mode'),
    'c': dict(color='brown', marker='s', label='c-mode'),
    'p': dict(color='darkorange',   marker='^', label='p-mode'),
}

# spin values used for the overlaid propagation diagram (view 3)
SPIN_OVERLAY = [-0.99, -0.5, 0.0, 0.5, 0.99]
SPIN_COLORS  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

print(f"Grid shape: {m_g.shape}  (m, n, a, r)")

def mode_regions(im, ij, tol_frac=0):
    """
    Return boolean arrays (NA, NR) for g, c, p regions.
    'Match' means: target frequency falls inside the region.
    tol_frac: fractional half-width around NU0 treated as a match band.
    """
    mm = int(round(m_g[im, ij, 0, 0]))
    nn = int(round(j_g[im, ij, 0, 0]))

    lo, hi = NU0 * (1 - tol_frac), NU0 * (1 + tol_frac)

    mO  = mm * Om[im, ij, :, :]          # (NA, NR)
    fpk = mO + Kp[im, ij, :, :]
    fpn = mO + Op[im, ij, :, :] * np.sqrt(nn)
    if mm > 0:
        fmk = mO - Kp[im, ij, :, :]
        fmn = mO - Op[im, ij, :, :] * np.sqrt(nn)
    else:
        fmk = fpk
        fmn = fpn

    if nn == 0:
        if mm == 0:
            kappa_1d = Kp[im, ij, :, :]
            kappa_max = np.nanmax(kappa_1d, axis=-1, keepdims=True)
            r_max_idx = np.nanargmax(kappa_1d, axis=-1)
            ia_idx = np.arange(kappa_1d.shape[0])
            r_max = RIVR[im, ij, ia_idx, r_max_idx]
            r_max = r_max[:, None]
            R_grid = RIVR[im, ij, :, :]
            p = (NU0 <= kappa_max) & (NU0 >= kappa_1d) & ~np.isnan(fpk) & (R_grid < r_max)
        else:
            p = (NU0 <= fmk) & ~np.isnan(fpk)
        c = np.zeros_like(p, dtype=bool)
        g = np.zeros_like(p, dtype=bool)

    elif mm == 0:
        kappa_2d = Kp[im, ij, :, :]
        g = (NU0 <= kappa_2d) & ~np.isnan(kappa_2d)
        c = np.zeros((NA, NR), dtype=bool)
        p = np.zeros((NA, NR), dtype=bool)

    else:
        g = (NU0 >= fmk) & (NU0 <= fpk) & ~np.isnan(fmk) & ~np.isnan(fpk)
        c = (NU0 <= fmn) & (fmn > 0) & ~np.isnan(fmn) & (NU0 > 0)
        p = np.zeros_like(g, dtype=bool)

    # remove g/c overlap with p
    g = g & ~p
    c = c & ~g & ~p

    return g, c, p, mm, nn


# ── resonance radius helpers ────────────────────────────────────────────
def r_ivr(a, nu_obs=NU0, m=1, n=1, M=M_BH, n_scan=8000):
    """
    Vertical resonance radius: ω̃ - sqrt(n)·Ω_z = 0
    i.e.  m·Ω_φ(r) - sqrt(n)·Ω_θ(r) = nu_obs
    Sign convention same as ILR/OLR.
    """
    a = float(a)
    isco = float(r_isco(a))
    r = np.geomspace(isco * 1.001, 5000.0, n_scan)

    om_tilde = 2*np.pi*nu_obs - m * 2*np.pi * nu_phi(r, a, M)
    oz       = 2*np.pi * nu_theta(r, a, M)

    # IVR: om_tilde + sqrt(n)*oz = 0
    diff = om_tilde + np.sqrt(n) * oz
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        return np.nan
    i = sign_changes[0]
    denom = diff[i+1] - diff[i]
    if denom == 0:
        return float(r[i])
    return float(r[i] - diff[i] * (r[i+1] - r[i]) / denom)

#############################

def resonance_curves(a_arr, nu_obs=NU0, m=1, n=1, M=M_BH):
    """Vectorised: return (r_ILR, r_OLR, r_IVR) arrays over a_arr."""
    r_ilr_arr = np.array([r_ilr(a, m, nu_obs, M) for a in a_arr])
    r_olr_arr = np.array([r_olr(a, m, nu_obs, M) for a in a_arr])
    r_ivr_arr = np.array([r_ivr(a, nu_obs, m, n, M) for a in a_arr])
    return r_ilr_arr, r_olr_arr, r_ivr_arr

##########################

# encode mode type as integer: 0=none, 1=g, 2=c, 3=p
CMAP_DISCRETE = mcolors.ListedColormap(['#F5F5F0', '#4E79A7', '#A0522D', '#E8A45A'])
BOUNDS = [-0.5, 0.5, 1.5, 2.5, 3.5]
NORM   = mcolors.BoundaryNorm(BOUNDS, CMAP_DISCRETE.N)

# coarser spin grid for resonance root-finding (one set per panel)
a_res = np.linspace(-0.998, 0.998, 60)

fig, axes = plt.subplots(NM, NJ, figsize=(7, 6.5), sharex=True, sharey=True,
                        gridspec_kw={"wspace": 0.05, "hspace": 0.05})

for im in range(NM):
    for ij in range(NJ):
        ax = axes[im, ij]
        fix_spines(ax)
        g, c, p, mm, nn = mode_regions(im, ij)

        code = np.zeros((NA, NR), dtype=float)
        code[g] = 1
        code[c] = 2
        code[p] = 3

        R_grid    = RIVR[im, ij, :, :]
        A_grid    = A[im, ij, :, :]
        isco_grid = r_isco(A_grid)
        code[R_grid < isco_grid] = np.nan

        ax.pcolormesh(
            A_grid, R_grid, code,
            cmap=CMAP_DISCRETE, norm=NORM,
            shading='auto', rasterized=True
        )

        # ISCO line
        ax.plot(a_vec, r_isco(a_vec), color='purple', ls='--', lw=1, alpha=0.7)

        # resonance curves — computed for this panel's (mm, nn)
        # ILR/OLR depend on m; IVR depends on m and n
        # skip degenerate m=0 case (no meaningful Lindblad resonances)
        if mm > 0:
            ilr_c, olr_c, ivr_c = resonance_curves(a_res, m=mm, n=max(nn, 1))
            ax.plot(a_res, ilr_c, color='#C0392B', lw=1.5, ls=':')
            ax.plot(a_res, olr_c, color='#C0392B', lw=1, ls='-.')
            if nn > 0:
                ax.plot(a_res, ivr_c, color='#00FF00', lw=1, ls='-')

        ax.set_yscale('log')
        ax.set_xlim(-1, 1)
        ax.set_ylim(1, 2e2)
        ax.grid(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune='both'))

        ax.text(0.1, 0.1, fr"$m={mm}, n={nn}$",
                #fontweight='bold',
                transform=ax.transAxes,
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.85))

        if im == NM-1: ax.set_xlabel(r"$a$")
        else:          ax.tick_params(labelbottom=False)
        if ij == 0:    ax.set_ylabel(r"$R\,[R_g]$")
        else:          ax.tick_params(labelleft=False)

# shared legend
p_none = mpatches.Patch(color='#F5F5F0', edgecolor='#999', label='no match')
p_g    = mpatches.Patch(color='#4E79A7', label='g-mode')
p_c    = mpatches.Patch(color='#A0522D', label='c-mode')
p_p    = mpatches.Patch(color='#E8A45A', label='p-mode')
h_i    = mlines.Line2D([],[],color='purple',   ls='--', lw=1, label='ISCO')
h_ilr  = mlines.Line2D([],[],color='#C0392B', ls=':',  lw=1.5, label='ILR')
h_olr  = mlines.Line2D([],[],color='#C0392B', ls='-.', lw=1, label='OLR')
h_ivr  = mlines.Line2D([],[],color='#D4AC0D', ls='-',  lw=1, label='IVR')
fig.legend(handles=[p_none, p_g, p_c, p_p, h_i, h_ilr, h_olr, h_ivr],
           loc='lower center', ncol=4, frameon=True,
           bbox_to_anchor=(0.5, -0.))

plt.tight_layout(rect=[0, 0.05, 1, 0.995])
plt.savefig('cmodes_plots/heatmap_rg.pdf', bbox_inches='tight')
plt.show()

