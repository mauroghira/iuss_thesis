"""
Summary constraints of the PIF model: maximal precession radius r_max(a) and
per-source constraints (table for the paper).

Key property used here (verified numerically on the whole spin grid, see
check_structure): for every spin, g = nu_solid * M decreases monotonically
with r_out, and the set of valid precession radii, i.e. those with

    N_w = nu_solid t_wave = g tau_w k_w < 1 < N_a = nu_solid t_align = g tau_a k_a,

is a single interval [r_isco, r_max(a)]. N_w and N_a do not depend on M, so
validity of a radius is mass independent; M and nu_0 only select which radius
precesses at nu_0, through g(a, r_bar) = P = M nu_0. The valid P window of a
spin is therefore [g(a, r_max), g(a, r_isco)].

Outputs (output_catalog/):
    pif_rmax.pdf                 r_max(a), fiducial and (H/R, alpha) envelope
    pif_source_constraints.tex   LaTeX table of per-source constraints
and a printed summary (joint constraints of sources with two QPOs, and the
dependence of each source's spin range on (H/R, alpha)).
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from setup import r_isco, set_style, fix_spines
from pif_core import ALPHA, HR_ISCO_DEFAULT
from catalog import CATALOG
from pif_mass_freq_map import build_spin_curves, make_spins, valid_on_P, R_OUT_GRID
from pif_timescales import resolve_mass_reference

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output_catalog')
FIG_FILE = os.path.join(OUT_DIR, 'pif_rmax.pdf')
TABLE_FILE = os.path.join(OUT_DIR, 'pif_source_constraints.tex')

A_THORNE = 0.998
SPINS = np.unique(np.r_[make_spins(), A_THORNE])
HR_VALS = (0.03, 0.1, 0.3)
ALPHA_VALS = (0.001, 0.01, 0.1)
N_M = 400                     # masses per source range


def rescaling(hr, al):
    """(k_w, k_a) multiplying tau_w, tau_a (pif_core.PIFGeometric.physical)."""
    return HR_ISCO_DEFAULT / hr, (hr / HR_ISCO_DEFAULT) ** 2 * (ALPHA / al)


def nearest(curves, spin):
    """Curve of the grid spin closest to `spin` (keys are floats)."""
    keys = np.array(list(curves))
    return curves[float(keys[np.argmin(np.abs(keys - spin))])]


def _usable(curve):
    g = curve.geo.g
    return np.isfinite(g).all() and np.any(g > 0)


def valid_radii(curve, k_w=1.0, k_a=1.0):
    """Boolean mask over curve.r_out: N_w < 1 < N_a (nodes of the r_out grid)."""
    g = curve.geo.g
    return (g * curve.geo.tau_w * k_w < 1.0) & (g * curve.geo.tau_a * k_a > 1.0)


def window(curve, k_w=1.0, k_a=1.0):
    """
    (r_max, limit, P_min, P_max) for one spin. r_max is located between the
    last valid node and the next one by log-log interpolation of the failing
    N (as in pif_mass_freq_map.classify_on_P); P_min = g(r_max).
    limit: the condition that fails just beyond r_max ('align', 'wave',
    'both', or 'edge' if r_max is the last node of the r_out grid).
    NaNs if no radius is valid.
    """
    g, r = curve.geo.g, curve.r_out
    v = valid_radii(curve, k_w, k_a)
    if not v.any():
        return np.nan, 'none', np.nan, np.nan
    k = np.flatnonzero(v)[-1]
    if k == r.size - 1:
        return r[k], 'edge', g[k], g[0]
    lNw = np.log(g[k:k + 2] * curve.geo.tau_w[k:k + 2] * k_w)    # <0 at k
    lNa = np.log(g[k:k + 2] * curve.geo.tau_a[k:k + 2] * k_a)    # >0 at k
    f_w = -lNw[0] / (lNw[1] - lNw[0]) if lNw[1] >= 0 else np.inf
    f_a = lNa[0] / (lNa[0] - lNa[1]) if lNa[1] <= 0 else np.inf
    f = min(f_w, f_a, 1.0)
    limit = 'both' if np.isclose(f_w, f_a) else ('wave' if f_w < f_a else 'align')
    lr = np.log(r[k]) + f * np.log(r[k + 1] / r[k])
    lg = np.log(g[k]) + f * np.log(g[k + 1] / g[k])
    return np.exp(lr), limit, np.exp(lg), g[0]


def check_structure(curves):
    """g monotonically decreasing and valid radii = one interval from r_isco."""
    bad = []
    for s, c in curves.items():
        if not _usable(c):
            continue
        idx = np.flatnonzero(valid_radii(c))
        mono = np.all(np.diff(c.geo.g) < 0)
        contiguous = idx.size == 0 or (idx[0] == 0 and np.all(np.diff(idx) == 1))
        if not (mono and contiguous):
            bad.append(s)
    return bad


def r_bar(curve, P):
    """Precession radius g(a, r_bar) = P (log-log interpolation, g monotonic)."""
    g, r = curve.geo.g, curve.r_out
    return np.exp(np.interp(np.log(P), np.log(g[::-1]), np.log(r[::-1]),
                            left=np.nan, right=np.nan))


# --------------------------------------------------------------------------
# Figure: r_max(a)
# --------------------------------------------------------------------------
def plot_rmax(curves, output=FIG_FILE):
    spins = np.array([s for s in SPINS if s in curves and _usable(curves[s])])
    fid = [window(curves[s]) for s in spins]
    r_fid = np.array([w[0] for w in fid])
    lim_fid = np.array([w[1] for w in fid])

    r_all = np.array([[window(curves[s], *rescaling(hr, al))[0] for s in spins]
                      for hr in HR_VALS for al in ALPHA_VALS])
    lo, hi = np.nanmin(r_all, axis=0), np.nanmax(r_all, axis=0)

    set_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    fix_spines(ax)
    ax.fill_between(spins, lo, hi, color='0.85', lw=0, zorder=1)

    styles = {'align': dict(color='#1f4e79', ls='-'),
              'wave': dict(color='#c0392b', ls='-'),
              'edge': dict(color='0.4', ls=':')}
    for lim, st in styles.items():
        y = np.where(lim_fid == lim, r_fid, np.nan)
        ax.plot(spins, y, lw=1.6, zorder=3, **st)

    ax.plot(spins, r_isco(spins), color='black', lw=0.8, ls='--', zorder=2)
    ax.plot(spins, r_isco(spins) * HR_ISCO_DEFAULT / ALPHA, color='black',
            lw=0.8, ls=(0, (1, 2)), zorder=2)

    ax.set_yscale('log')
    ax.set_xlim(-1, 1)
    ax.set_ylim(1, R_OUT_GRID[-1])
    ax.set_xlabel(r'Spin $a$')
    ax.set_ylabel(r'$\bar r$ [$R_g$]')

    handles = [
        Line2D([], [], lw=1.6, **styles['align'],
               label=r'$r_\mathrm{max}$: $1/\nu_0<t_\mathrm{align}$ fails beyond'),
        Line2D([], [], lw=1.6, **styles['wave'],
               label=r'$r_\mathrm{max}$: $t_\mathrm{wave}<1/\nu_0$ fails beyond'),
        Patch(color='0.85', label=r'$r_\mathrm{max}$ for $h_\mathrm{ISCO}\in[0.03,0.3]$, '
                                  r'$\alpha\in[10^{-3},10^{-1}]$'),
        Line2D([], [], color='black', lw=0.8, ls='--', label=r'$r_\mathrm{ISCO}$'),
        Line2D([], [], color='black', lw=0.8, ls=(0, (1, 2)),
               label=r'$r_\mathrm{ISCO}\,h_\mathrm{ISCO}/\alpha$ (fiducial)'),
    ]
    fig.legend(handles=handles, loc='outside lower center', ncol=2,
               fontsize=6, frameon=True)
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return output


# --------------------------------------------------------------------------
# Per-source constraints
# --------------------------------------------------------------------------
def source_constraints(source, curves, k_w=1.0, k_a=1.0, mass_range=None):
    """
    Valid (M, a) cells of a source within mass_range (default: catalogue range;
    a point estimate is used as such). Returns a dict with spin ranges by sign,
    r_bar range, the maximal mass allowed by the model for any spin, and the
    fraction of the (log) mass range where at least one spin is valid.
    """
    nu0 = source['nu_rest']
    lo, hi = mass_range or source['mass_range']
    masses = np.array([lo]) if lo == hi else np.logspace(np.log10(lo), np.log10(hi), N_M)
    P = masses * nu0

    spins, ok, rb = [], [], []
    for s in SPINS:
        c = curves.get(float(s))
        if c is None or not _usable(c):
            continue
        v = valid_on_P(c, P, k_w, k_a)
        spins.append(s)
        ok.append(v)
        rb.append(np.where(v, r_bar(c, P), np.nan))
    spins, ok, rb = np.array(spins), np.array(ok), np.array(rb)   # (n_a, n_M)

    def srange(sel):
        a = spins[sel & ok.any(axis=1)]
        return (a.min(), a.max()) if a.size else None

    P_max_any = window(nearest(curves, A_THORNE), k_w, k_a)[3]
    return {
        'P': (P.min(), P.max()),
        'a_pos': srange(spins > 0),
        'a_neg': srange(spins < 0),
        'r_bar': (np.nanmin(rb), np.nanmax(rb)) if ok.any() else None,
        'M_max': P_max_any / nu0,
        'frac_M': ok.any(axis=0).mean(),
        'ok': ok, 'spins': spins, 'masses': masses,
    }


def _fmt_spin(a):
    return f'{a:+.3f}' if np.isclose(a, A_THORNE) else f'{a:+.2f}'


def _fmt_range(r):
    if r is None:
        return '--'
    a, b = r
    if np.isclose(a, b):
        return f'${_fmt_spin(a)}$'
    return f'${_fmt_spin(a)}$ to ${_fmt_spin(b)}$'


def _fmt_P(P):
    a, b = np.log10(P)
    return f'${a:.1f}$' if np.isclose(a, b) else f'${a:.1f}$--${b:.1f}$'


def write_table(rows, output=TABLE_FILE):
    lines = [
        r'\begin{table*}',
        r'\caption{Constraints on the individual sources for the fiducial parameters '
        r'($h_\textup{ISCO}=0.1$, $\alpha=0.01$).}',
        r'\label{tab:constraints}',
        r'\centering',
        r'\begin{tabular}{lccccc}',
        r'\hline\hline',
        r'Source & $\log P$ & $a>0$ & $a<0$ & $\bar r$ [$R_g$] & '
        r'$\log M_\textup{max}$ \\',
        r'\hline',
    ]
    for name, c in rows:
        rb = '--' if c['r_bar'] is None else f"${c['r_bar'][0]:.1f}$--${c['r_bar'][1]:.1f}$"
        cut = r'$^\dagger$' if c['frac_M'] < 1 else ''
        lines.append(
            f"{name} & {_fmt_P(c['P'])} & {_fmt_range(c['a_pos'])} & "
            f"{_fmt_range(c['a_neg'])} & {rb} & ${np.log10(c['M_max']):.1f}${cut} \\\\")
    lines += [
        r'\hline',
        r'\end{tabular}',
        r'\tablefoot{',
        r'$P=M\nu_0(1+z)$ in Hz\,$M_\odot$, with the rest-frame frequency, over the mass range '
        r'of Table~\ref{tab:catalog} (at the point estimate for NGC~4945 and J1257).',
        r'Columns 3--5: range of prograde and retrograde spins, and of precession radii, '
        r'for which at least one mass in the range is valid (spin step 0.005).',
        r'$M_\textup{max}$ in $M_\odot$: largest mass with a valid configuration, '
        r'$M_\textup{max}=P_\textup{max}/[\nu_0(1+z)]$, where '
        r'$P_\textup{max}=|\nu_\textup{LT}(r_\textup{ISCO})|\,m\simeq1.05\times10^4$\,Hz\,$M_\odot$ '
        r'is the largest $P$ allowed for any spin, reached for $a=0.998$ and $\bar r\to r_\textup{ISCO}$; '
        r'it does not depend on $h_\textup{ISCO}$ and $\alpha$.',
        r'$^\dagger$~$M_\textup{max}$ is below the upper end of the mass range.',
        r'}',
        r'\end{table*}',
    ]
    with open(output, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return output


PAIRS = [(4, 5), (6, 7), (10, 11), (12, 13)]


def joint(c1, c2):
    """Joint (M, a) validity of two QPOs of the same source (same grids)."""
    J = c1['ok'] & c2['ok']
    if not J.any():
        return None
    a = c1['spins'][J.any(axis=1)]
    M = c1['masses'][J.any(axis=0)]
    return (a.min(), a.max()), (M.min(), M.max())


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    curves = build_spin_curves(SPINS)
    bad = check_structure(curves)
    print('structure check (monotonic g, valid radii = [r_isco, r_max]):',
          'OK' if not bad else f'FAILS for spins {bad}')

    print('\nr_max(a), fiducial:')
    for s in (-1, -0.5, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, A_THORNE):
        rm, lim, Pmin, Pmax = window(nearest(curves, s))
        print(f'  a={s:+.3f} r_isco={r_isco(s):.2f} r_max={rm:6.1f} '
              f'({rm / r_isco(s):.2f} r_isco, {lim:5s}) P=[{Pmin:.3g}, {Pmax:.3g}]')
    print('Saved:', plot_rmax(curves))

    by_idx = {s['idx']: s for s in CATALOG}
    cons = {i: source_constraints(by_idx[i], curves) for i in by_idx}
    print('Saved:', write_table([(by_idx[i]['name'], cons[i]) for i in by_idx]))

    print('\nJoint constraints (two QPOs, same M and a):')
    for i, j in PAIRS:
        res = joint(cons[i], cons[j])
        print(f"  {by_idx[i]['name']} + {by_idx[j]['name']}:",
              'none' if res is None else
              f'a {res[0][0]:+.3f}..{res[0][1]:+.3f}, M {res[1][0]:.2e}..{res[1][1]:.2e}')

    print('\nSpin ranges vs (h_ISCO, alpha):')
    for i in by_idx:
        print(f"  {by_idx[i]['name']}")
        for hr in HR_VALS:
            for al in ALPHA_VALS:
                c = source_constraints(by_idx[i], curves, *rescaling(hr, al))
                print(f'    h={hr:<5} alpha={al:<6} a>0 {_fmt_range(c["a_pos"]):22s} '
                      f'a<0 {_fmt_range(c["a_neg"]):22s} f_M={c["frac_M"]:.2f}')
