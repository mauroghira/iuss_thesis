"""
Dependence of the PIF validity on the radial profile of the thickness,

    H / R = h_ISCO (r / r_ISCO)^(1/2 - q)        (pif_core.sound_speed).

q = 3/2 is the fiducial choice of the paper (H independent of radius, Motta
et al. 2017); q = 1/2 is a flow with constant aspect ratio, H / R = h_ISCO at
every radius (geometrically thick hot flow). Unlike h_ISCO and alpha, q does
not enter through a constant factor, so the geometric curves must be
recomputed (pif_mass_freq_map.build_spin_curves with q = Q_ALT); the rescaling
with (h_ISCO, alpha) of pif_hr_alpha.rescaling stays exact for every q.

For q = 1/2 the valid region reaches much larger radii than for q = 3/2, so
the outer radii extend to R_OUT_EXT_MAX = 1000 R_g (R_OUT_GRID_EXT), with the
radial step of the integrals kept below 0.1 R_g (radial_nodes).

Outputs:
    plots_paper/pif_P_spin_map_q05.pdf    (a, P) plane, fiducial h_ISCO, alpha
    plots_paper/pif_hr_alpha_map_q05.pdf  (a, P) plane on the (h_ISCO, alpha) grid
    output_q/*.npz                        data of the two figures
and a printed summary (windows, spin of XRBs, N_a, individual sources), from
which the numbers quoted in the paper are taken.

Usage: python pif_q.py            (figures from the cached data, if present)
       python pif_q.py --recompute
"""

import os
import sys

import numpy as np

from setup import r_isco, g_LT
from catalog import CATALOG
from pif_core import ALPHA, HR_ISCO_DEFAULT, Q_SOUND
from pif_mass_freq_map import (build_spin_curves, compute_map, plot_P_spin,
                               R_OUT_GRID_EXT)
import pif_constraints as pc
import pif_hr_alpha as pha

Q_ALT = 0.5

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output_q')
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots_paper')
MAP_FILE = os.path.join(OUT_DIR, 'pif_mass_frequency_map_q05.npz')
GRID_FILE = os.path.join(OUT_DIR, 'pif_hr_alpha_q05.npz')
FIG_P_FILE = os.path.join(PLOT_DIR, 'pif_P_spin_map_q05.pdf')
FIG_GRID_FILE = os.path.join(PLOT_DIR, 'pif_hr_alpha_map_q05.pdf')

TABLE_SPINS = (-1, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95,
               0.98, pc.A_THORNE)
# Spin of XRBs: reflection gives a > 0.9 in several sources with type-C QPOs
XRB_SPINS = (0.9, 0.95, 0.98, pc.A_THORNE)
XRB_P = (1.0, 10.0, 100.0)        # 0.1, 1, 10 Hz for M = 10 Msun
# (h_ISCO, alpha) for the tables besides the fiducial pair: a hot flow
HOT_FLOWS = ((0.2, 0.01), (0.2, 0.1), (0.3, 0.1))
N_CYCLES = (2, 5, 10, 100)


def g_max(a):
    return float(abs(g_LT(r_isco(a), a)))


def loglog_at(curve, values, radius):
    """Log-log interpolation of values(curve.r_out) at radius."""
    return float(np.exp(np.interp(np.log(radius), np.log(curve.r_out),
                                  np.log(values))))


def numbers(curve):
    """N_w = g tau_w, N_a = g tau_a at the reference (h_ISCO, alpha)."""
    return curve.geo.g * curve.geo.tau_w, curve.geo.g * curve.geo.tau_a


def valid_intervals(curve, k_w=1.0, k_a=1.0):
    """[(r_lo, r_hi), ...]: runs of consecutive valid nodes of the r_out grid."""
    v = pc.valid_radii(curve, k_w, k_a)
    edges = np.flatnonzero(np.diff(np.r_[False, v, False].astype(int)))
    return [(curve.r_out[i], curve.r_out[j - 1])
            for i, j in zip(edges[::2], edges[1::2])]


def radius_of_cycles(curve, n, k_a=1.0):
    """Smallest radius at which N_a = n (N_a decreases outwards); NaN if
    N_a > n on the whole grid."""
    r, g = curve.r_out, curve.geo.g
    d = np.log(g * curve.geo.tau_a * k_a) - np.log(n)
    idx = np.flatnonzero(d[:-1] * d[1:] <= 0)
    if idx.size == 0:
        return np.nan, np.nan
    i = idx[0]
    f = -d[i] / (d[i + 1] - d[i])
    return (np.exp(np.log(r[i]) + f * np.log(r[i + 1] / r[i])),
            np.exp(np.log(g[i]) + f * np.log(g[i + 1] / g[i])))


# ------------------------------------------------------------------ summary
def print_windows(curves, label, combos=((HR_ISCO_DEFAULT, ALPHA),)):
    for hr, al in combos:
        k = pc.rescaling(hr, al)
        print(f'\nValid window, {label}, h={hr:g}, alpha={al:g} '
              '(radii in R_g, P in Hz Msun):')
        for s in TABLE_SPINS:
            c = pc.nearest(curves, s)
            rm, lim, Pmin, _ = pc.window(c, *k)
            runs = valid_intervals(c, *k)
            gap = '' if len(runs) <= 1 else (
                '  NOT one interval: valid for r in '
                + ', '.join(f'[{a:.2f}, {b:.2f}]' for a, b in runs))
            op = '>' if lim == 'edge' else '='
            print(f'  a={s:+.3f} r_isco={r_isco(s):.2f} r_max{op}{rm:7.1f} '
                  f'({rm / r_isco(s):6.1f} r_isco) P_min{"<" if lim == "edge" else "="}'
                  f'{Pmin:9.3g} g_max={g_max(s):9.4g} '
                  f'width={np.log10(g_max(s) / Pmin):.2f} dex {lim}{gap}')


def print_wave(curves, label):
    """Smallest h_ISCO for which waves cross the flow at every radius."""
    print(f'\nWave propagation, {label}: largest N_w at h={HR_ISCO_DEFAULT:g} '
          'and minimum h for N_w < 1 at all radii of the grid')
    for s in (0.5, 0.7, 0.9, 0.95, 0.98, 0.99, pc.A_THORNE):
        c = pc.nearest(curves, s)
        Nw, _ = numbers(c)
        print(f'  a={s:.3f} max N_w={Nw.max():.3f} at r={c.r_out[Nw.argmax()]:.2f} '
              f'({c.r_out[Nw.argmax()] / r_isco(s):.2f} r_isco) -> '
              f'h > {HR_ISCO_DEFAULT * Nw.max():.3f}')


def print_xrb(curves, label, hr_for_alpha):
    """Bounds on (h_ISCO, alpha) for a rapidly spinning hole at the P of
    type-C QPOs: N_w (0.1 / h) < 1 and N_a (h / 0.1)^2 (0.01 / alpha) > 1."""
    print(f'\nSpin of XRBs, {label}: precession radius, minimum h_ISCO (waves) '
          f'and maximum alpha (alignment) at h = {hr_for_alpha}')
    for s in XRB_SPINS:
        c = pc.nearest(curves, s)
        Nw, Na = numbers(c)
        for P in XRB_P:
            rb = pc.r_bar(c, np.array([P]))[0]
            if not np.isfinite(rb):
                print(f'  a={s:.3f} P={P:5g}: no precession radius on the grid')
                continue
            nw, na = loglog_at(c, Nw, rb), loglog_at(c, Na, rb)
            h_min = HR_ISCO_DEFAULT * nw
            a_max = [ALPHA * (h / HR_ISCO_DEFAULT) ** 2 * na for h in hr_for_alpha]
            print(f'  a={s:.3f} P={P:5g} ({P / 10:g} Hz at 10 Msun): '
                  f'r_bar={rb:6.1f} ({rb / r_isco(s):5.1f} r_isco) '
                  f'N_w={nw:.3g} N_a={na:.3g} h_min={h_min:.3f} alpha_max='
                  + ', '.join(f'{x:.2g}' for x in a_max))


def print_cycles(curves, label):
    print(f'\nN_a, {label}, fiducial h, alpha: r/r_isco and log(g_max/P) at '
          f'N_a = {N_CYCLES} (-- : N_a larger on the whole grid; '
          '* : beyond r_max, set by waves)')
    for s in (-1, -0.5, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98):
        c = pc.nearest(curves, s)
        rm = pc.window(c)[0]
        out = []
        for n in N_CYCLES:
            r, g = radius_of_cycles(c, n)
            out.append('   --      ' if not np.isfinite(r) else
                       f'{r / r_isco(s):6.1f}{"*" if r > rm else " "}/{np.log10(g_max(s) / g):.2f}')
        print(f'  a={s:+.2f} ' + ' '.join(out))


def print_sources(curves_fid, curves_alt, combos):
    by_idx = {s['idx']: s for s in CATALOG}
    print('\nIndividual sources: valid spins and precession radii over the '
          'mass range')
    for i, src in by_idx.items():
        print(f"  {src['name']}")
        rows = [(f'q={Q_SOUND:g} h={HR_ISCO_DEFAULT:g} al={ALPHA:g}',
                 pc.source_constraints(src, curves_fid))]
        rows += [(f'q={Q_ALT:g} h={hr:g} al={al:g}',
                  pc.source_constraints(src, curves_alt, *pc.rescaling(hr, al)))
                 for hr, al in combos]
        for tag, c in rows:
            rb = '--' if c['r_bar'] is None else f"{c['r_bar'][0]:.1f}-{c['r_bar'][1]:.1f}"
            print(f'    {tag:22s} a>0 {pc._fmt_range(c["a_pos"]):22s} '
                  f'a<0 {pc._fmt_range(c["a_neg"]):22s} r_bar {rb:12s} '
                  f'f_M={c["frac_M"]:.2f}')
    print('\nJoint constraints (two QPOs, same M and a):')
    for i, j in pc.PAIRS:
        for tag, cv, k in ([(f'q={Q_SOUND:g}', curves_fid, (1.0, 1.0))]
                           + [(f'q={Q_ALT:g} h={hr:g} al={al:g}', curves_alt,
                               pc.rescaling(hr, al)) for hr, al in combos]):
            res = pc.joint(pc.source_constraints(by_idx[i], cv, *k),
                           pc.source_constraints(by_idx[j], cv, *k))
            print(f"  {by_idx[i]['name']} + {by_idx[j]['name']} [{tag}]:",
                  'none' if res is None else
                  f'a {res[0][0]:+.3f}..{res[0][1]:+.3f}, '
                  f'M {res[1][0]:.2e}..{res[1][1]:.2e}')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    recompute = '--recompute' in sys.argv

    # ---- figures
    if recompute or not os.path.exists(MAP_FILE):
        compute_map(q=Q_ALT, r_out_grid=R_OUT_GRID_EXT, data_file=MAP_FILE)
    plot_P_spin(MAP_FILE, overlay_sources=False, fig_file=FIG_P_FILE)
    if recompute or not os.path.exists(GRID_FILE):
        pha.compute(GRID_FILE, q=Q_ALT, r_out_grid=R_OUT_GRID_EXT)
    pha.plot_P_map(pha.load(GRID_FILE), outfile=FIG_GRID_FILE)

    # ---- numbers
    curves_fid = build_spin_curves(pc.SPINS)
    curves_alt = build_spin_curves(pc.SPINS, q=Q_ALT, r_out_grid=R_OUT_GRID_EXT)
    fid = f'q={Q_SOUND:g}'
    alt = f'q={Q_ALT:g}'
    combos = ((HR_ISCO_DEFAULT, ALPHA),) + HOT_FLOWS

    print_windows(curves_fid, fid)
    print_windows(curves_alt, alt, combos)
    print_wave(curves_fid, fid)
    print_wave(curves_alt, alt)
    print_xrb(curves_fid, fid, hr_for_alpha=(0.1, 0.2, 0.3))
    print_xrb(curves_alt, alt, hr_for_alpha=(0.1, 0.2, 0.3))
    print_cycles(curves_fid, fid)
    print_cycles(curves_alt, alt)
    print_sources(curves_fid, curves_alt, combos)
