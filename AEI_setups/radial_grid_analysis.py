"""
radial_grid_analysis.py
=======================
Analisi radiale su griglia di parametri — versione aggiornata.

Compatibile con la nuova API di aei_common.compute_disk_profile:
  • disk_model è passato esplicitamente come argomento a radial_scan_grid
  • params è un dict {'a':..., 'B00':..., 'Sigma0':...}
  • nessun monkey-patch necessario per cambiare modello

Funzioni principali
-------------------
  radial_scan_grid        itera su griglia (a, B00, Sigma0), raccoglie profili
                          radiali completi e li aggrega in bin radiali
  plot_radial_grid        4 pannelli: k·r, β, dQ/dr, frac_AEI  vs r
                          mediana ± IQR per zona, aggregati sulla griglia
  plot_validity_heatmap   heatmap 2D della frazione AEI valida nello spazio
                          dei parametri, per range radiale e zona scelti
  compute_slopes_grid     esponenti delle leggi di potenza per (a, B00, Σ₀, zona)
  plot_slope_distributions violinplot + scatter degli esponenti per zona
  summary_table_grid      tabella riassuntiva per zona (%, slopes mediane)

Dipendenze: aei_common.py, setup.py
           (NON dipende più direttamente da full_disk_SS)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize

import sys
sys.path.append("..")
from setup import create_param_grid, M_BH

from .aei_common import compute_disk_profile, HOR, mm as MM_DEFAULT

# colori e nomi zone — definiti qui in modo autonomo
ZONE_NAMES  = ['A', 'B', 'C']
ZONE_COLORS = {'A': '#f97316', 'B': '#3b82f6', 'C': '#22c55e'}
_COLOR_FALLBACK = '#94a3b8'   # grigio — per zone 'N/A' o sconosciute


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SCAN SU GRIGLIA
# ═══════════════════════════════════════════════════════════════════════════════

def radial_scan_grid(
    param_dict,
    disk_model,
    mm=MM_DEFAULT,
    hr=HOR,
    n_r=150,
    r_max=None,
    quantities=('k', 'beta', 'dQdr'),
    n_rbins=30,
    M=M_BH,
    verbose=True,
):
    """
    Itera su una griglia (a, B00, Sigma0) e per ogni combinazione calcola
    il profilo radiale completo tramite compute_disk_profile, poi aggrega
    i risultati in bin radiali log-spaziati.

    Parameters
    ----------
    param_dict : dict
        Formato: {'a': (min,max,n), 'B00': (min,max,n), 'Sigma0': (min,max,n)}
        Usa la stessa convenzione di create_param_grid del notebook.

    disk_model : callable
        disk_model(r_rg, **params) → (B0, Sigma, c_s, zone[, info])
        Stessa firma richiesta da find_rossby e compute_disk_profile.
        Esempi:
          lambda r, **p: disk_model_SS(r, **p, alpha_visc=0.01, hr=0.05)
          lambda r, **p: disk_model_NT(r, **p, alpha_visc=0.01, hr=0.05)
          lambda r, **p: disk_model_simple(r, **p, hr=0.05)

    mm : int
        Modo azimutale m della perturbazione AEI.

    hr : float
        Aspect ratio H/r.

    n_r : int
        Numero di punti radiali per ogni profilo individuale.

    r_max : float o None
        Raggio esterno della griglia [r_g].
        Se None: ricavato automaticamente da info['r_BC'] × 3 se disponibile,
                 altrimenti 1000 r_g come fallback.

    quantities : tuple of str
        Colonne del DataFrame da aggregare.
        Valori possibili: 'k', 'beta', 'dQdr', 'B0', 'Sigma', 'c_s'.

    n_rbins : int
        Numero di bin radiali log-spaziati per l'aggregazione.

    M : float
        Massa del buco nero [M_sun].

    verbose : bool
        Stampa progressi e statistiche.

    Returns
    -------
    df_all : pd.DataFrame
        Tutti i punti radiali di tutti i run concatenati.
        Colonne garantite: r, zone, B0, Sigma, c_s, k, beta, dQdr,
                           k_valid, beta_valid, shear_valid, aei_valid,
                           a, B00, Sigma0.

    df_binned : pd.DataFrame
        Statistiche per bin radiale aggregate su tutta la griglia.
        Colonne: r_mid, zone, {qty}_median, {qty}_q1, {qty}_q3,
                 {qty}_mean, {qty}_std, count, frac_k, frac_beta,
                 frac_shear, frac_aei.

    meta_list : list of dict
        Metadati (r_AB, r_BC, r_ISCO, mdot, …) per ogni run.
    """
    vectors = create_param_grid(param_dict, mesh=False)
    a_vals   = vectors['a']
    B00_vals = vectors['B00']
    S0_vals  = vectors['Sigma0']

    total = len(a_vals) * len(B00_vals) * len(S0_vals)
    if verbose:
        print(f"Grid scan: {len(a_vals)} × {len(B00_vals)} × {len(S0_vals)} "
              f"= {total} combinazioni")

    all_frames = []
    meta_list  = []
    done = 0

    for a_val in a_vals:
        for B00_val in B00_vals:
            for S0_val in S0_vals:
                params = {'a': a_val, 'B00': B00_val, 'Sigma0': S0_val}
                try:
                    df_run, meta = compute_disk_profile(
                        disk_model = disk_model,
                        params     = params,
                        mm         = mm,
                        hr         = hr,
                        M          = M,
                        n_points   = n_r,
                        r_max      = r_max,   # None → auto via info['r_BC']
                    )

                    df_run['a']      = a_val
                    df_run['B00']    = B00_val
                    df_run['Sigma0'] = S0_val
                    all_frames.append(df_run)
                    meta_list.append(meta)

                except Exception as e:
                    if verbose:
                        print(f"  skip a={a_val:.2f} B={B00_val:.1e} "
                              f"S={S0_val:.1e}: {e}")

                done += 1
                if verbose and done % max(1, total // 10) == 0:
                    print(f"  {done}/{total}  ({done/total*100:.0f}%)")

    if not all_frames:
        raise RuntimeError("Nessun profilo calcolato — controlla i parametri.")

    df_all = pd.concat(all_frames, ignore_index=True)

    # ── bin radiali log-spaziati sull'intero range di r ──────────────────────
    r_lo = df_all['r'].min()
    r_hi = df_all['r'].max()
    edges  = np.geomspace(r_lo, r_hi, n_rbins + 1)
    r_mids = np.sqrt(edges[:-1] * edges[1:])

    df_all['r_bin'] = pd.cut(df_all['r'], bins=edges, labels=r_mids)
    df_all['r_bin'] = df_all['r_bin'].astype(float)

    # ── aggregazione per (r_bin, zone) ───────────────────────────────────────
    # Usa le zone effettivamente presenti — funziona con 'A','B','C' (SS/NT)
    # e con 'N/A' (simple_disc)
    zones_present = df_all['zone'].unique().tolist()
    records = []
    for zone in zones_present:
        sub_z = df_all[df_all['zone'] == zone]
        for r_mid in r_mids:
            sub = sub_z[sub_z['r_bin'] == r_mid]
            if len(sub) < 3:
                continue
            row = {'r_mid': r_mid, 'zone': zone, 'count': len(sub)}
            for qty in quantities:
                if qty not in sub.columns:
                    continue
                col = sub[qty].dropna()
                if qty == 'dQdr':
                    row[f'{qty}_median'] = col.median()
                    row[f'{qty}_q1']     = col.quantile(0.25)
                    row[f'{qty}_q3']     = col.quantile(0.75)
                    row[f'{qty}_mean']   = col.mean()
                    row[f'{qty}_std']    = col.std()
                else:
                    pos = col[col > 0]
                    if len(pos) < 2:
                        continue
                    lp = np.log10(pos)
                    row[f'{qty}_median'] = 10**lp.median()
                    row[f'{qty}_q1']     = 10**lp.quantile(0.25)
                    row[f'{qty}_q3']     = 10**lp.quantile(0.75)
                    row[f'{qty}_mean']   = 10**lp.mean()
                    row[f'{qty}_std']    = lp.std()

            N = len(sub)
            row['frac_k']     = sub['k_valid'].sum()     / N
            row['frac_beta']  = sub['beta_valid'].sum()  / N
            row['frac_shear'] = sub['shear_valid'].sum() / N
            row['frac_aei']   = sub['aei_valid'].sum()   / N
            records.append(row)

    df_binned = pd.DataFrame(records)

    if verbose:
        print(f"\nPunti totali: {len(df_all)}")
        print(f"Bin radiali con dati: {len(df_binned)}")
        _print_zone_summary(df_all)

    return df_all, df_binned, meta_list


def _print_zone_summary(df_all):
    """Stampa veloce della frazione AEI per zona."""
    print("\n  Zona  |  N punti  | % k ok | % β≤1  | % shear | % AEI")
    print("  " + "-"*55)
    for zone in df_all['zone'].unique():
        sub = df_all[df_all['zone'] == zone]
        N = len(sub)
        if N == 0:
            continue
        pk = sub['k_valid'].sum()    / N * 100
        pb = sub['beta_valid'].sum() / N * 100
        ps = sub['shear_valid'].sum()/ N * 100
        pa = sub['aei_valid'].sum()  / N * 100
        print(f"    {zone}   | {N:>9} | {pk:>5.1f}% | {pb:>5.1f}% | {ps:>6.1f}% | {pa:>5.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PLOT PROFILI RADIALI BINNED  (mediana ± IQR)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_radial_grid(df_binned, quantities=('k', 'beta', 'dQdr'),
                    figsize=(16, 12), title=""):
    """
    Pannelli radiali: mediana ± IQR per zona aggregata sulla griglia.

    Pannelli fissi: k, β, dQ/dr, frazione AEI valida (sempre il quarto).
    Qualsiasi quantità aggiuntiva viene aggiunta come pannello extra.

    Parameters
    ----------
    df_binned : DataFrame da radial_scan_grid
    quantities : tuple of str
        Quantità da mostrare nei pannelli (escluso il pannello frac_AEI
        che è sempre aggiunto come ultimo).
    figsize : tuple
    title : str

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_panels = len(quantities) + 1   # +1 per il pannello frazioni
    ncols = min(n_panels, 2)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize[0], figsize[1] * nrows / 2),
                             squeeze=False)
    axes_flat = axes.flatten()
    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    _log_qty  = {'k', 'beta', 'B0', 'Sigma', 'c_s'}
    _ylabel   = {
        'k':     'k  (adimensionale)',
        'beta':  'β  (plasma beta)',
        'dQdr':  'dQ/dr  [u.a.]',
        'B0':    'B₀  [G]',
        'Sigma': 'Σ  [g/cm²]',
        'c_s':   'c_s  [cm/s]',
    }
    _hrefs = {
        'k':     [(0.1, 'gray', ':'), (10.0, 'gray', ':')],
        'beta':  [(1.0, 'red',  '--')],
        'dQdr':  [(0.0, 'red',  '--')],
    }

    for ax, qty in zip(axes_flat, quantities):
        logscale = qty in _log_qty
        zones_present = df_binned['zone'].unique()
        for zone in zones_present:
            col = ZONE_COLORS.get(zone, _COLOR_FALLBACK)
            sub = df_binned[df_binned['zone'] == zone].sort_values('r_mid')
            if sub.empty or f'{qty}_median' not in sub.columns:
                continue
            med = sub[f'{qty}_median']
            q1  = sub[f'{qty}_q1']
            q3  = sub[f'{qty}_q3']
            label = f'Zona {zone}' if zone != 'N/A' else 'disco'
            ax.plot(sub['r_mid'], med, color=col, lw=2, label=label)
            ax.fill_between(sub['r_mid'], q1, q3, color=col, alpha=0.18)

        for (yval, hcol, hls) in _hrefs.get(qty, []):
            ax.axhline(yval, color=hcol, ls=hls, lw=1, alpha=0.7)

        ax.set_xscale('log')
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel('r [rg]', fontsize=11)
        ylabel = _ylabel.get(qty, qty)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=11)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=9)

    # ── pannello frazioni (sempre l'ultimo) ──────────────────────────────────
    ax_frac = axes_flat[len(quantities)]
    for zone in df_binned['zone'].unique():
        col = ZONE_COLORS.get(zone, _COLOR_FALLBACK)
        sub = df_binned[df_binned['zone'] == zone].sort_values('r_mid')
        if sub.empty:
            continue
        label = f'Zona {zone}' if zone != 'N/A' else 'disco'
        ax_frac.plot(sub['r_mid'], sub['frac_aei'],  color=col, lw=2,
                     label=f'{label} — AEI')
        ax_frac.plot(sub['r_mid'], sub['frac_k'],    color=col, lw=1,
                     ls='--', alpha=0.55)
        ax_frac.plot(sub['r_mid'], sub['frac_beta'], color=col, lw=1,
                     ls=':',  alpha=0.55)

    ax_frac.set_xscale('log')
    ax_frac.set_ylim(0, 1.05)
    ax_frac.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.7)
    ax_frac.set_xlabel('r [rg]', fontsize=11)
    ax_frac.set_ylabel('Frazione valida', fontsize=11)
    ax_frac.set_title('Frazione AEI valida', fontsize=11)
    ax_frac.grid(True, alpha=0.15)
    ax_frac.plot([], [], 'k-',  lw=2,   label='AEI (tutti)')
    ax_frac.plot([], [], 'k--', lw=1, alpha=0.55, label='k fisico')
    ax_frac.plot([], [], 'k:',  lw=1, alpha=0.55, label='β ≤ 1')
    ax_frac.legend(fontsize=8)

    # nascondi pannelli in eccesso
    for ax in axes_flat[len(quantities) + 1:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  HEATMAP  frac_aei  nello spazio dei parametri
# ═══════════════════════════════════════════════════════════════════════════════

def plot_validity_heatmap(df_all, param_x='B00', param_y='Sigma0',
                          r_range=(None, None), zone=None,
                          metric='aei_valid',
                          n_bins_x=15, n_bins_y=15, figsize=(10, 7)):
    """
    Heatmap 2D: asse x = param_x, asse y = param_y,
    colore = metrica aggregata nel range radiale e zona selezionati.

    Parameters
    ----------
    df_all   : DataFrame da radial_scan_grid
    param_x  : str  colonna asse x  ('a', 'B00', 'Sigma0')
    param_y  : str  colonna asse y  ('a', 'B00', 'Sigma0')
    r_range  : (r_min, r_max) | (None, None)  — None = tutto il range
    zone     : str | None  — filtra per zona ('A', 'B', 'C')
    metric   : str  — colonna booleana da aggregare come media
               ('aei_valid', 'k_valid', 'beta_valid', 'shear_valid')
    n_bins_x, n_bins_y : int  — risoluzione della griglia 2D
    figsize  : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    sub = df_all.copy()
    if zone:
        sub = sub[sub['zone'] == zone]
    r_lo = r_range[0] if r_range[0] is not None else sub['r'].min()
    r_hi = r_range[1] if r_range[1] is not None else sub['r'].max()
    sub  = sub[(sub['r'] >= r_lo) & (sub['r'] <= r_hi)]

    if sub.empty:
        print("Nessun dato nel range / zona selezionati.")
        return None

    log_x = param_x in ('B00', 'Sigma0')
    log_y = param_y in ('B00', 'Sigma0')

    x_vals = np.log10(sub[param_x].values.astype(float)) if log_x else sub[param_x].values.astype(float)
    y_vals = np.log10(sub[param_y].values.astype(float)) if log_y else sub[param_y].values.astype(float)
    m_vals = sub[metric].values.astype(float)

    x_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins_x + 1)
    y_edges = np.linspace(y_vals.min(), y_vals.max(), n_bins_y + 1)

    grid_val = np.full((n_bins_y, n_bins_x), np.nan)

    for ix in range(n_bins_x):
        for iy in range(n_bins_y):
            mask = ((x_vals >= x_edges[ix]) & (x_vals < x_edges[ix + 1]) &
                    (y_vals >= y_edges[iy]) & (y_vals < y_edges[iy + 1]))
            if mask.sum() >= 3:
                grid_val[iy, ix] = m_vals[mask].mean()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid_val, origin='lower', aspect='auto',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   vmin=0, vmax=1, cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label=f'<{metric}>')

    _label = {
        'B00':    'log₁₀(B₀₀ [G])',
        'Sigma0': 'log₁₀(Σ₀ [g/cm²])',
        'a':      'spin  a',
    }
    ax.set_xlabel(_label.get(param_x, param_x), fontsize=12)
    ax.set_ylabel(_label.get(param_y, param_y), fontsize=12)

    zone_str = f"Zona {zone}" if zone else "Tutte le zone"
    ax.set_title(
        f"{metric} — {zone_str}\n"
        f"r ∈ [{r_lo:.1f}, {r_hi:.1f}] rg",
        fontsize=12
    )
    ax.grid(False)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ESPONENTI DELLE LEGGI DI POTENZA
# ═══════════════════════════════════════════════════════════════════════════════

def _pl_slope(x, y):
    """Fit log-log lineare → pendenza, o NaN se dati insufficienti."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan
    lx = np.log10(x[mask])
    ly = np.log10(y[mask])
    mx, my = lx.mean(), ly.mean()
    denom = ((lx - mx) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((lx - mx) * (ly - my)).sum() / denom)


def compute_slopes_grid(df_all):
    """
    Per ogni run (a, B00, Sigma0) e ogni zona calcola le pendenze
    delle leggi di potenza  B∝r^α,  Σ∝r^α,  β∝r^α,  k·r∝r^α.

    Parameters
    ----------
    df_all : DataFrame da radial_scan_grid

    Returns
    -------
    df_slopes : DataFrame
        Colonne: a, B00, Sigma0, zone, slope_B, slope_S, slope_beta
    """
    records = []
    zones_present = df_all['zone'].unique().tolist()   # ← usa le zone reali
    for (a_val, B00_val, S0_val), grp in df_all.groupby(['a', 'B00', 'Sigma0']):
        for zone in zones_present:                     # ← non ZONE_NAMES fisso
            sub = grp[grp['zone'] == zone]
            if len(sub) < 5:
                continue
            r = sub['r'].values
            rec = {
                'a':         a_val,
                'B00':       B00_val,
                'Sigma0':    S0_val,
                'zone':      zone,
                'slope_B':   _pl_slope(r, sub['B0'].values)    if 'B0'    in sub else np.nan,
                'slope_S':   _pl_slope(r, sub['Sigma'].values) if 'Sigma' in sub else np.nan,
                'slope_beta':_pl_slope(r, sub['beta'].values)  if 'beta'  in sub else np.nan,
            }
            records.append(rec)
    return pd.DataFrame(records)


def plot_slope_distributions(df_slopes, figsize=(16, 10)):
    """
    Violinplot + scatter degli esponenti delle leggi di potenza per zona.
    Il colore dei punti rappresenta lo spin a.

    Parameters
    ----------
    df_slopes : DataFrame da compute_slopes_grid
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    quantities = [
        ('slope_B',    'B ∝ r^α   →   α'),
        ('slope_S',    'Σ ∝ r^α   →   α'),
        ('slope_beta', 'β ∝ r^α   →   α'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=False)
    fig.suptitle("Distribuzioni degli esponenti delle leggi di potenza", fontsize=13)

    for ax, (col, label) in zip(axes, quantities):
        zones_in_data = df_slopes['zone'].unique().tolist()
        data_by_zone = [
            df_slopes[df_slopes['zone'] == z][col].dropna().values
            for z in zones_in_data        # ← non ZONE_NAMES
        ]
        valid = [d for d in data_by_zone if len(d) > 1]
        if valid:
            parts = ax.violinplot(
                valid,
                positions=range(len(valid)),
                showmedians=True, showextrema=True
            )
            for pc, zone in zip(parts['bodies'], ZONE_NAMES):
                pc.set_facecolor(ZONE_COLORS[zone])
                pc.set_alpha(0.45)

        sc_handle = None
        for i, zone in enumerate(ZONE_NAMES):
            sub = df_slopes[df_slopes['zone'] == zone].dropna(subset=[col])
            if sub.empty:
                continue
            sc_handle = ax.scatter(
                np.random.normal(i, 0.04, len(sub)),
                sub[col].values,
                c=sub['a'].values, cmap='RdBu', vmin=-1, vmax=1,
                s=14, alpha=0.65, zorder=3
            )

        ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.set_xticks(range(len(ZONE_NAMES)))
        ax.set_xticklabels([f'Zona {z}' for z in ZONE_NAMES])
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label.split('→')[0].strip(), fontsize=11)
        ax.grid(True, alpha=0.15, axis='y')

    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=Normalize(-1, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[-1], label='Spin  a', shrink=0.8)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TABELLA RIASSUNTIVA GRIGLIA
# ═══════════════════════════════════════════════════════════════════════════════

def summary_table_grid(df_all, df_binned=None, df_slopes=None):
    """
    Stampa statistiche aggregate per zona sulla griglia completa.

    Parameters
    ----------
    df_all     : DataFrame da radial_scan_grid
    df_binned  : DataFrame (opzionale, per info aggiuntive sui bin)
    df_slopes  : DataFrame da compute_slopes_grid (opzionale)

    Returns
    -------
    df_summary : pd.DataFrame  con le stesse info stampate
    """
    n_runs = df_all.groupby(['a', 'B00', 'Sigma0']).ngroups
    print("\n" + "=" * 78)
    print("  ANALISI RADIALE — SINTESI GRIGLIA")
    print("=" * 78)
    print(f"  Run totali: {n_runs}   |   Punti totali: {len(df_all)}")
    print(f"  Range r:    [{df_all['r'].min():.1f}, {df_all['r'].max():.0f}] rg")
    if df_binned is not None:
        print(f"  Bin radiali con dati: {len(df_binned)}")
    print()

    has_slopes = df_slopes is not None
    header = (f"{'Zona':>5}  {'N':>9}  {'% k':>6}  {'% β':>6}  "
              f"{'% sh':>6}  {'% AEI':>7}")
    if has_slopes:
        header += f"  {'α_B':>8}  {'α_Σ':>8}  {'α_β':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for zone in ZONE_NAMES:
        sub = df_all[df_all['zone'] == zone]
        N = len(sub)
        if N == 0:
            continue
        pct = lambda c: sub[c].sum() / N * 100

        row_dict = {
            'zone':       zone,
            'N':          N,
            'pct_k':      pct('k_valid'),
            'pct_beta':   pct('beta_valid'),
            'pct_shear':  pct('shear_valid'),
            'pct_aei':    pct('aei_valid'),
        }
        line = (f"    {zone}  {N:>9}  {pct('k_valid'):>5.1f}%  "
                f"{pct('beta_valid'):>5.1f}%  {pct('shear_valid'):>5.1f}%  "
                f"{pct('aei_valid'):>6.1f}%")

        if has_slopes:
            sz = df_slopes[df_slopes['zone'] == zone]
            for sc in ['slope_B', 'slope_S', 'slope_beta']:
                v = sz[sc].median() if not sz.empty else np.nan
                row_dict[sc] = v
                line += f"  {v:>8.3f}"

        print(line)
        rows.append(row_dict)

    print("=" * 78)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CONFRONTO TRA MODELLI DIVERSI
# ═══════════════════════════════════════════════════════════════════════════════

def compare_models(results_dict, figsize=(16, 10)):
    """
    Confronta i risultati di scan su modelli diversi sullo stesso grafico.

    Parameters
    ----------
    results_dict : dict  { 'nome_modello': (df_all, df_binned, meta_list) }
        Dizionario con i risultati di radial_scan_grid per ogni modello.
        Esempio:
          {
            'SS':     (df_all_ss,  df_binned_ss,  meta_ss),
            'NT':     (df_all_nt,  df_binned_nt,  meta_nt),
            'Simple': (df_all_s,   df_binned_s,   meta_s),
          }
    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    linestyles = ['-', '--', ':', '-.']
    quantities = ['k', 'beta', 'dQdr']
    n_panels = len(quantities) + 1   # +1 per il pannello frazioni
    ncols = min(n_panels, 2)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle("Confronto modelli — mediana su tutta la griglia", fontsize=13)

    _log_qty = {'k', 'beta', 'B0', 'Sigma', 'c_s'}
    _ylabel  = {'k': 'k', 'beta': 'β', 'dQdr': 'dQ/dr'}
    _hrefs   = {
        'k':    [(0.1, 'gray', ':'), (10, 'gray', ':')],
        'beta': [(1.0, 'red', '--')],
        'dQdr': [(0.0, 'red', '--')],
    }

    cmap  = plt.cm.tab10
    model_colors = {name: cmap(i) for i, name in enumerate(results_dict)}

    for ax, qty in zip(axes[:len(quantities)], quantities):
        logscale = qty in _log_qty
        for (name, (df_all, df_binned, _)), ls in zip(results_dict.items(), linestyles):
            col = model_colors[name]
            for zone in ZONE_NAMES:
                sub = df_binned[df_binned['zone'] == zone].sort_values('r_mid')
                if sub.empty or f'{qty}_median' not in sub.columns:
                    continue
                alpha = 1.0 if zone == 'B' else 0.5
                ax.plot(sub['r_mid'], sub[f'{qty}_median'],
                        color=col, ls=ls, lw=1.5 + (zone == 'B') * 0.5,
                        alpha=alpha, label=f'{name}·{zone}' if zone == 'A' else '_')

        for (yval, hcol, hls) in _hrefs.get(qty, []):
            ax.axhline(yval, color=hcol, ls=hls, lw=1, alpha=0.6)

        ax.set_xscale('log')
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel('r [rg]', fontsize=11)
        ax.set_ylabel(_ylabel.get(qty, qty), fontsize=11)
        ax.set_title(_ylabel.get(qty, qty), fontsize=11)
        ax.grid(True, alpha=0.15)

    # ── pannello frac_aei per ogni modello (tutte le zone aggregate) ─────────
    ax_frac = axes[-1]
    for (name, (df_all, df_binned, _)), ls in zip(results_dict.items(), linestyles):
        col = model_colors[name]
        # aggrega su tutte le zone: per ogni r_mid prendi la media pesata
        sub_all = df_binned.groupby('r_mid', as_index=False).agg(
            frac_aei_mean=('frac_aei', 'mean')
        ).sort_values('r_mid')
        ax_frac.plot(sub_all['r_mid'], sub_all['frac_aei_mean'],
                     color=col, ls=ls, lw=2, label=name)

    ax_frac.set_xscale('log')
    ax_frac.set_ylim(0, 1.05)
    ax_frac.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.6)
    ax_frac.set_xlabel('r [rg]', fontsize=11)
    ax_frac.set_ylabel('Frazione AEI valida', fontsize=11)
    ax_frac.set_title('Frac AEI (media zone)', fontsize=11)
    ax_frac.legend(fontsize=9)
    ax_frac.grid(True, alpha=0.15)

    # legenda modelli su primo pannello
    handles = [plt.Line2D([0], [0], color=model_colors[n], ls=ls, lw=2, label=n)
               for n, ls in zip(results_dict, linestyles)]
    axes[0].legend(handles=handles, fontsize=9, title='Modello')

    plt.tight_layout()
    return fig
