"""
pif_timescales_catalog.py

  Pannello 1: M = M_ref fissata (dal mass_range del catalogo, o valore
      generico AGN se assente), spin variabile -> analogo di Fig. 7
      di Motta et al. 2018.
  Pannello 2: a = A_FIXED fissato, M variabile nel range appropriato
      alla sorgente (banda riportata, range esplorativo attorno a una
      stima puntuale, o range AGN generico).

In ogni caso NU_TARGET = nu0 della sorgente 

Gestione di mass_range secondo il docstring di catalog.py:
  - None                -> nessuna stima indipendente. Pannello 1: M_ref
                            = media geometrica di (M_AGN_MIN, M_AGN_MAX).
                            Pannello 2: scan sull'intero range generico.
  - (lo, hi) con lo==hi  -> stima puntuale. Pannello 1: M_ref = lo.
                            Pannello 2: scan ESPLORATIVO di +-1 dex
                            attorno al valore puntuale (assunzione
                            dichiarata: NON e' un intervallo di
                            confidenza sulla stima, va letto solo come
                            sensitivity check).
  - (lo, hi) con lo<hi   -> banda riportata in tesi. Pannello 1: M_ref
                            = media geometrica di (lo, hi). Pannello 2:
                            scan esattamente su (lo, hi).
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from setup import (r_isco, set_style, fix_spines, M_AGN_MIN, M_AGN_MAX,
                   nu_solid_vect)
from disk_profiles import t_wave_closed
from align_timescale import t_align_vect
from catalog import CATALOG, select_sources

set_style()

# --- parametri fissi (assunzioni dichiarate, come nello script originale) ---
ALPHA = 0.01
A_VALS = np.array([-0.9, -0.5, -0.25, 0.0, 0.01, 0.25, 0.5, 0.9, 0.998])   # pannello 1
A_FIXED = 0.5                                              # pannello 2
N_M = 6                                                     # n. curve M nel pannello 2
N_A_SUMMARY = 100                                            # spin per il riepilogo
N_M_SUMMARY = 50                                             # masse per il riepilogo
R_OUT_GRID = np.logspace(0, np.log10(200.0), 100)
EXPLORATORY_DEX = 1.0   # +-1 ordine di grandezza attorno a stima puntuale


def resolve_mass_reference(mass_range):
    """
    Da mass_range del catalogo ricava:
      M_ref  : massa singola per il pannello 1
      M_lo, M_hi : estremi dello scan per il pannello 2
      kind   : 'point' | 'band' | 'generic' | 'exploratory'
                (per annotare correttamente il pannello 2)
    """
    if mass_range is None:
        M_lo, M_hi = M_AGN_MIN, M_AGN_MAX
        M_ref = np.sqrt(M_lo * M_hi)
        return M_ref, M_lo, M_hi, 'generic'

    lo, hi = mass_range
    if lo == hi:
        M_ref = lo
        M_lo = lo / 10**EXPLORATORY_DEX
        M_hi = lo * 10**EXPLORATORY_DEX
        return M_ref, M_lo, M_hi, 'exploratory'

    M_ref = np.sqrt(lo * hi)
    return M_ref, lo, hi, 'band'


def safe_filename(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')


def _crossing_radii(radii, values, target):
    valid = np.isfinite(values) & (values > 0)
    radii = radii[valid]
    values = values[valid]
    if radii.size < 2:
        return []

    log_delta = np.log(values) - np.log(target)
    crossing_indices = np.flatnonzero(log_delta[:-1] * log_delta[1:] <= 0)
    crossings = []
    for index in crossing_indices:
        left_delta = log_delta[index]
        right_delta = log_delta[index + 1]
        if left_delta == 0:
            radius = radii[index]
        elif right_delta == 0:
            radius = radii[index + 1]
        else:
            fraction = -left_delta / (right_delta - left_delta)
            log_radius = np.log(radii[index]) + fraction * (
                np.log(radii[index + 1]) - np.log(radii[index])
            )
            radius = np.exp(log_radius)
        crossings.append(radius)
    return crossings


def _mark_precession_crossings(ax, radii, a, r_in, mass, nu_target, color):
    a_arr = np.full_like(radii, a)
    r_in_arr = np.full_like(radii, r_in)
    mass_arr = np.full_like(radii, mass)
    zeta_arr = np.zeros_like(radii)
    nu_solid = nu_solid_vect(a_arr, r_in_arr, radii, zeta_arr, mass_arr)

    crossings = _crossing_radii(radii, nu_solid, nu_target)
    for radius in crossings:
        ax.plot(radius, 1.0 / nu_target, marker='o', ms=4, color=color,
                markeredgecolor='black', markeredgewidth=0.35,
                linestyle='None', zorder=5)
    return crossings


def _panel1_fixed_mass(ax, M_ref, nu_target):
    matches = []
    colors_a = plt.cm.viridis(np.linspace(0, 1, len(A_VALS)))
    for a, col in zip(A_VALS, colors_a):
        r_in = r_isco(a)
        mask = R_OUT_GRID > r_in * 1.001
        r_out_v = R_OUT_GRID[mask]
        if r_out_v.size == 0:
            continue
        r_in_arr = np.full_like(r_out_v, r_in)
        a_arr = np.full_like(r_out_v, a)
        M_arr = np.full_like(r_out_v, M_ref)

        t_wave = t_wave_closed(r_in_arr, r_out_v, a_arr, M_arr)
        t_align = t_align_vect(a_arr, r_in_arr, r_out_v, M_arr,
                                np.full(r_out_v.size, ALPHA))

        ax.plot(r_out_v, t_wave, color=col, lw=1, ls=':')
        ax.plot(r_out_v, t_align, color=col, lw=1, ls='--',
                 label=fr"$a={a:.2f}$")
        r_precession = _mark_precession_crossings(
            ax, r_out_v, a, r_in, M_ref, nu_target, col)
        matches.append({
            'label': fr'$a={a:.2f}$',
            'r_precession': r_precession,
            'r_wave': _crossing_radii(r_out_v, t_wave, 1.0 / nu_target),
            'r_align': _crossing_radii(r_out_v, t_align, 1.0 / nu_target),
        })

    ax.axhline(1.0 / nu_target, color='black', ls='-', lw=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, R_OUT_GRID[-1])
    ax.set_xlabel(r"$R_{\rm out}$ [$R_g$]")
    ax.set_ylabel("Tempo [s]")
    ax.legend(loc='upper right', frameon=True, fontsize=6.5, ncol=2)
    return matches


def _panel2_fixed_spin(ax, M_lo, M_hi, nu_target):
    matches = []
    r_in_fixed = r_isco(A_FIXED)
    mask2 = R_OUT_GRID > r_in_fixed * 1.001
    r_out_v2 = R_OUT_GRID[mask2]

    M_vals = np.logspace(np.log10(M_lo), np.log10(M_hi), N_M)
    colors_M = plt.cm.plasma(np.linspace(0, 1, len(M_vals)))

    for M, col in zip(M_vals, colors_M):
        r_in_arr = np.full_like(r_out_v2, r_in_fixed)
        a_arr = np.full_like(r_out_v2, A_FIXED)
        M_arr = np.full_like(r_out_v2, M)

        t_wave = t_wave_closed(r_in_arr, r_out_v2, a_arr, M_arr)
        t_align = t_align_vect(a_arr, r_in_arr, r_out_v2, M_arr,
                                np.full(r_out_v2.size, ALPHA))

        ax.plot(r_out_v2, t_wave, color=col, lw=1, ls=':')
        ax.plot(r_out_v2, t_align, color=col, lw=1, ls='--',
                 label=fr"$M={M:.1e}\,M_\odot$")
        r_precession = _mark_precession_crossings(
            ax, r_out_v2, A_FIXED, r_in_fixed, M, nu_target, col)
        matches.append({
            'label': fr'$M={M:.1e}\,M_\odot$',
            'r_precession': r_precession,
            'r_wave': _crossing_radii(r_out_v2, t_wave, 1.0 / nu_target),
            'r_align': _crossing_radii(r_out_v2, t_align, 1.0 / nu_target),
        })

    ax.axhline(1.0 / nu_target, color='black', ls='-', lw=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, R_OUT_GRID[-1])
    ax.set_xlabel(r"$R_{\rm out}$ [$R_g$]")
    ax.legend(loc='upper right', frameon=True, fontsize=6.5, ncol=2)
    return matches


def _format_radii(radii):
    if not radii:
        return '--'
    return ', '.join(f'{radius:.2f}' for radius in radii)


def _summary_rows(matches):
    rows = []
    for match in matches:
        r_precession = match['r_precession']
        r_wave = match['r_wave']
        r_align = match['r_align']
        timescale_matches = [radii[0] for radii in (r_wave, r_align) if radii]
        if not timescale_matches:
            status = 'OK (no crossing)'
        elif r_precession:
            status = 'OK' if r_precession[0] < min(timescale_matches) else 'NO'
        else:
            status = '--'
        rows.append([
            match['label'],
            _format_radii(r_precession),
            _format_radii(r_wave),
            _format_radii(r_align),
            status,
        ])
    return rows


def _match_is_ok(match):
    r_precession = match['r_precession']
    timescale_matches = [
        radii[0] for radii in (match['r_wave'], match['r_align']) if radii
    ]
    if not timescale_matches:
        return True
    return bool(r_precession) and r_precession[0] < min(timescale_matches)


def _source_matches(source, a_values=A_VALS, n_m=N_M):
    nu0 = source['nu0']
    M_ref, M_lo, M_hi, _ = resolve_mass_reference(source['mass_range'])
    matches1 = []
    for a in a_values:
        r_in = r_isco(a)
        r_out = R_OUT_GRID[R_OUT_GRID > r_in * 1.001]
        if r_out.size == 0:
            continue
        r_in_arr = np.full_like(r_out, r_in)
        a_arr = np.full_like(r_out, a)
        M_arr = np.full_like(r_out, M_ref)
        t_wave = t_wave_closed(r_in_arr, r_out, a_arr, M_arr)
        t_align = t_align_vect(
            a_arr, r_in_arr, r_out, M_arr, np.full(r_out.size, ALPHA)
        )
        r_precession = _crossing_radii(
            r_out,
            nu_solid_vect(a_arr, r_in_arr, r_out, np.zeros_like(r_out), M_arr),
            nu0,
        )
        matches1.append({
            'label': fr'$a={a:.2f}$',
            'parameter': a,
            'parameter_kind': 'a',
            'r_precession': r_precession,
            'r_wave': _crossing_radii(r_out, t_wave, 1.0 / nu0),
            'r_align': _crossing_radii(r_out, t_align, 1.0 / nu0),
        })

    r_in = r_isco(A_FIXED)
    r_out = R_OUT_GRID[R_OUT_GRID > r_in * 1.001]
    matches2 = []
    for mass in np.logspace(np.log10(M_lo), np.log10(M_hi), n_m):
        r_in_arr = np.full_like(r_out, r_in)
        a_arr = np.full_like(r_out, A_FIXED)
        M_arr = np.full_like(r_out, mass)
        t_wave = t_wave_closed(r_in_arr, r_out, a_arr, M_arr)
        t_align = t_align_vect(
            a_arr, r_in_arr, r_out, M_arr, np.full(r_out.size, ALPHA)
        )
        r_precession = _crossing_radii(
            r_out,
            nu_solid_vect(a_arr, r_in_arr, r_out, np.zeros_like(r_out), M_arr),
            nu0,
        )
        matches2.append({
            'label': fr'$M={mass:.1e}\,M_\odot$',
            'parameter': mass,
            'parameter_kind': 'M',
            'r_precession': r_precession,
            'r_wave': _crossing_radii(r_out, t_wave, 1.0 / nu0),
            'r_align': _crossing_radii(r_out, t_align, 1.0 / nu0),
        })
    return matches1, matches2


def _format_ok_ranges(matches):
    if not matches:
        return '--'

    values = np.array([match['parameter'] for match in matches])
    ok = np.array([_match_is_ok(match) for match in matches])
    if not np.any(ok):
        return '--'

    ranges = []
    ok_transitions = np.diff(ok.astype(int))
    starts = np.flatnonzero(np.r_[ok[0], ok_transitions == 1])
    ends = np.flatnonzero(np.r_[ok_transitions == -1, ok[-1]])
    parameter_kind = matches[0]['parameter_kind']
    for start, end in zip(starts, ends):
        first = values[start]
        last = values[end]
        if parameter_kind == 'a':
            if start == end:
                ranges.append(f'a={first:.3f}')
            else:
                ranges.append(f'a in [{first:.3f}, {last:.3f}]')
        elif start == end:
            ranges.append(f'M={first:.2e}')
        else:
            ranges.append(f'M in [{first:.2e}, {last:.2e}]')
    return '; '.join(ranges)


def plot_catalog_summary(indices=None, outdir='.'): 
    """Save one table summarising OK conditions for selected catalog sources."""
    table_rows = []
    dense_spins = np.linspace(-1.0, 1.0, N_A_SUMMARY + 2)[1:-1]
    sources = select_sources(indices) if indices is not None else CATALOG
    for source in sources:
        matches1, matches2 = _source_matches(
            source, a_values=dense_spins, n_m=N_M_SUMMARY
        )
        matches = matches1 + matches2
        ok_matches = [match for match in matches if _match_is_ok(match)]
        M_ref, _, _, _ = resolve_mass_reference(source['mass_range'])
        spin_ranges = _format_ok_ranges(matches1)
        mass_ranges = _format_ok_ranges(matches2)
        spin_ranges = (
            rf'{spin_ranges} ($M_{{\rm fixed}}={M_ref:.2e}\,M_\odot$)'
        )
        mass_ranges = f'{mass_ranges}'
        table_rows.append([
            source['name'],
            'YES' if ok_matches else 'NO',
            f'{len(ok_matches)}/{len(matches)}',
            spin_ranges,
            mass_ranges,
        ])
        print(f"{source['name']}: spins={spin_ranges}; masses={mass_ranges}")

    fig = plt.figure(figsize=(14, max(4.0, 0.42 * (len(table_rows) + 2))))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.90])
    ax.axis('off')
    ax.set_title(
        fr'Catalog summary: conditions for solid precesion to occur',
        pad=12,
    )
    table = ax.table(
        cellText=table_rows,
        colLabels=['source', 'any OK', 'valid matches',
               r'OK spin ranges ($M_{\rm fixed}$)',
               r'OK mass ranges ($a_{\rm fixed} = {A_FIXED:.2f}$)'],
        cellLoc='center', colLoc='center', bbox=[0, 0, 1, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(col=list(range(5)))
    fname = os.path.join(outdir, 'pif_timescales_catalog_summary.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    return fname


def plot_source_timescales(source, outdir='.'):
    """
    Genera e salva la figura a due pannelli per una singola sorgente del
    catalogo. Restituisce il path del file salvato.
    """
    nu0 = source['nu0']
    name = source['name']
    M_ref, M_lo, M_hi, kind = resolve_mass_reference(source['mass_range'])

    kind_label = {
        'generic':     "M generica AGN (nessuna stima indipendente)",
        'point':       "M stima puntuale",
        'band':        "M da banda riportata in tesi",
        'exploratory': fr"M stima puntuale $\pm${EXPLORATORY_DEX:.0f} dex (esplorativo)",
    }[kind]

    fig = plt.figure(figsize=(10.5, 8.5))
    ax1 = fig.add_axes([0.08, 0.39, 0.40, 0.54])
    ax2 = fig.add_axes([0.56, 0.39, 0.40, 0.54], sharey=ax1)
    fix_spines(ax1)
    fix_spines(ax2)

    matches1 = _panel1_fixed_mass(ax1, M_ref, nu0)
    ax1.set_title(fr"$M_{{\rm ref}} = {M_ref:.2e}\,M_\odot$ fissata"
                   "\n" + kind_label, fontsize=8)

    matches2 = _panel2_fixed_spin(ax2, M_lo, M_hi, nu0)
    ax2.set_title(fr"$a={A_FIXED:.2f}$ fissato, $M \in [{M_lo:.1e}, {M_hi:.1e}]\,M_\odot$",
                   fontsize=8)

    style_handles = [
        Line2D([], [], color='gray', lw=1, ls=':', label=r'$t_{\rm wave}$'),
        Line2D([], [], color='gray', lw=1, ls='--', label=r'$t_{\rm align}$'),
        Line2D([], [], color='black', lw=1.5, ls='-', label=r'$P_{\rm osc}=1/\nu_0$'),
        Line2D([], [], color='black', marker='o', ms=4, linestyle='None',
               markeredgewidth=0.35, label=r'$\nu_{\rm solid}=\nu_0$'),
    ]
    table_rows = _summary_rows(matches1 + matches2)
    fig.suptitle(fr"{name}  —  $\nu_0 = {nu0:.2e}$ Hz", fontsize=10)
    fig.legend(handles=style_handles, loc='lower center', ncol=4,
               frameon=True, fontsize=7, bbox_to_anchor=(0.5, 0.335))
    table_ax = fig.add_axes([0.02, 0.04, 0.96, 0.24])
    table_ax.axis('off')
    table = table_ax.table(
        cellText=table_rows,
        colLabels=['parameter', r'$r_{\rm solid}$', r'$r_{\rm wave}$',
                   r'$r_{\rm align}$', 'result'],
        cellLoc='center', colLoc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)

    fname = os.path.join(outdir, f"pif_timescales_{safe_filename(name)}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    return fname


def plot_catalog(indices=None, outdir='.'):
    """
    Genera le figure per un sottoinsieme del catalogo (o per tutto il
    catalogo se indices=None), NELL'ORDINE dato da indices (se fornito),
    altrimenti nell'ordine canonico del catalogo.
    """
    sources = select_sources(indices) if indices is not None else CATALOG
    saved = []
    for source in sources:
        fname = plot_source_timescales(source, outdir=outdir)
        print(f"Salvato: {fname}")
        saved.append(fname)
    return saved


if __name__ == "__main__":
    os.makedirs("output_catalog", exist_ok=True)
    plot_catalog(indices=[0,9,10,11], outdir="output_catalog")
    summary_fname = plot_catalog_summary(indices=list(range(12)),
                                         outdir="output_catalog")
    print(f"Salvato: {summary_fname}")