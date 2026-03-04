"""
radial_grid_analysis.py
=======================
Estensione di full_disk_SS.py per l'analisi radiale su griglia di parametri.

Funzioni principali:
  - radial_scan_grid     : itera su griglia (a, B00, Sigma0) e raccoglie
                           profili radiali binned per zona
  - plot_radial_grid     : grafici mediana ± IQR per k, β, dQ/dr vs r
  - plot_validity_heatmap: heatmap 2D della frazione AEI valida nello spazio
                           dei parametri, per bin radiale scelto
  - plot_slope_grid      : distribuzioni degli esponenti delle leggi di potenza
                           al variare dei parametri

Dipende da: full_disk_SS.py, setup.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
from itertools import product

from setup import create_param_grid, r_isco, M_BH
from full_disk_SS import (
    compute_full_disk_profile,
    ZONE_NAMES, ZONE_COLORS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SCAN SU GRIGLIA
# ═══════════════════════════════════════════════════════════════════════════════

def radial_scan_grid(param_dict, n_r=150, r_max_factor=2.5,
                     quantities=('k', 'kr', 'beta', 'dQdr'),
                     n_rbins=30, M=M_BH, verbose=True):
    """
    Itera su una griglia (a, B00, Sigma0) e per ogni combinazione calcola
    il profilo radiale completo, poi lo aggrega in bin radiali.

    Parameters
    ----------
    param_dict : dict
        Formato: {'a': (min,max,n), 'B00': (min,max,n), 'Sigma0': (min,max,n)}
        Usa la stessa convenzione di create_param_grid del notebook.
    n_r : int
        Punti radiali per ogni profilo individuale.
    r_max_factor : float
        r_max = r_max_factor * r_BC per ogni profilo.
    quantities : tuple of str
        Colonne del DataFrame da aggregare ('k','kr','beta','dQdr').
    n_rbins : int
        Numero di bin radiali log-spaziati per l'aggregazione.
    verbose : bool
        Stampa progresso.

    Returns
    -------
    df_all : DataFrame
        Tutti i punti radiali di tutti i run concatenati.
        Colonne: r, zone, k, kr, beta, dQdr, k_valid, beta_valid,
                 shear_valid, aei_valid, a, B00, Sigma0.
    df_binned : DataFrame
        Statistiche per bin radiale (mediana, Q1, Q3, mean, std, count)
        aggregate su tutta la griglia.
        Colonne: r_mid, zone, {qty}_median, {qty}_q1, {qty}_q3,
                 {qty}_mean, {qty}_std, count, frac_k, frac_beta,
                 frac_shear, frac_aei.
    meta_list : list of dict
        Metadati (r_AB, r_BC, ecc.) per ogni run.
    """
    # costruisci vettori 1D
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
                try:
                    df_run, meta = compute_full_disk_profile(
                        a=a_val, B00=B00_val, Sigma0=S0_val,
                        M=M, n_points=n_r, r_max_factor=r_max_factor,
                        check_norm=False
                    )
                    df_run['a']      = a_val
                    df_run['B00']    = B00_val
                    df_run['Sigma0'] = S0_val
                    all_frames.append(df_run)
                    meta_list.append(meta)
                except Exception as e:
                    if verbose:
                        print(f"  skip a={a_val:.2f} B={B00_val:.1e} S={S0_val:.1e}: {e}")

                done += 1
                if verbose and done % max(1, total // 10) == 0:
                    print(f"  {done}/{total} ({done/total*100:.0f}%)")

    if not all_frames:
        raise RuntimeError("Nessun profilo calcolato — controlla i parametri.")

    df_all = pd.concat(all_frames, ignore_index=True)

    # ── bin radiali log-spaziati sull'intero range di r ──────────────────────
    r_min = df_all['r'].min()
    r_max = df_all['r'].max()
    edges = np.geomspace(r_min, r_max, n_rbins + 1)
    r_mids = np.sqrt(edges[:-1] * edges[1:])

    df_all['r_bin'] = pd.cut(df_all['r'], bins=edges, labels=r_mids)
    df_all['r_bin'] = df_all['r_bin'].astype(float)

    # ── aggregazione per (r_bin, zone) ───────────────────────────────────────
    records = []
    for zone in ZONE_NAMES:
        sub_z = df_all[df_all['zone'] == zone]
        for r_mid in r_mids:
            sub = sub_z[sub_z['r_bin'] == r_mid]
            if len(sub) < 3:
                continue
            row = {'r_mid': r_mid, 'zone': zone, 'count': len(sub)}
            for qty in quantities:
                col = sub[qty].dropna()
                if qty == 'dQdr':
                    # non log → stats lineari
                    row[f'{qty}_median'] = col.median()
                    row[f'{qty}_q1']     = col.quantile(0.25)
                    row[f'{qty}_q3']     = col.quantile(0.75)
                    row[f'{qty}_mean']   = col.mean()
                    row[f'{qty}_std']    = col.std()
                else:
                    # log-stats per quantità positive
                    pos = col[col > 0]
                    if len(pos) < 2:
                        continue
                    lp = np.log10(pos)
                    row[f'{qty}_median'] = 10**lp.median()
                    row[f'{qty}_q1']     = 10**lp.quantile(0.25)
                    row[f'{qty}_q3']     = 10**lp.quantile(0.75)
                    row[f'{qty}_mean']   = 10**lp.mean()
                    row[f'{qty}_std']    = lp.std()   # in dex

            # frazioni di validità
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

    return df_all, df_binned, meta_list


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PLOT PROFILI RADIALI BINNED  (mediana ± IQR)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_radial_grid(df_binned, figsize=(16, 12), title=""):
    """
    4 pannelli: k·r, β, dQ/dr, frazione AEI valida vs r.
    Per ogni zona mostra mediana e banda IQR aggregati sulla griglia.
    """
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=13)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    panels = [
        ('kr',    'k·r  (adimensionale)',    True,  [(0.1,'gray',':'), (10,'gray',':')]),
        ('beta',  'β  (plasma beta)',         True,  [(1.0,'red','--')]),
        ('dQdr',  'dQ/dr  [u.a.]',           False, [(0,'red','--')]),
        (None,    'Frazione AEI valida',      False, [(0.5,'gray',':')]),
    ]

    for ax, (qty, ylabel, logscale, hrefs) in zip(axes, panels):
        for zone in ZONE_NAMES:
            col  = ZONE_COLORS[zone]
            sub  = df_binned[df_binned['zone'] == zone].sort_values('r_mid')
            if sub.empty:
                continue

            if qty is None:
                # pannello frazioni
                ax.plot(sub['r_mid'], sub['frac_aei'],  color=col, lw=2,   label=f'{zone} AEI')
                ax.plot(sub['r_mid'], sub['frac_k'],    color=col, lw=1, ls='--', alpha=0.5)
                ax.plot(sub['r_mid'], sub['frac_beta'], color=col, lw=1, ls=':',  alpha=0.5)
                ax.set_ylim(0, 1.05)
            else:
                med = sub[f'{qty}_median']
                q1  = sub[f'{qty}_q1']
                q3  = sub[f'{qty}_q3']
                ax.plot(sub['r_mid'], med, color=col, lw=2, label=f'Zona {zone}')
                ax.fill_between(sub['r_mid'], q1, q3, color=col, alpha=0.18)

        for (yval, hcol, hls) in hrefs:
            ax.axhline(yval, color=hcol, ls=hls, lw=1, alpha=0.7)

        ax.set_xscale('log')
        if logscale:
            ax.set_yscale('log')
        ax.set_xlabel('r [rg]', fontsize=11)
        ax.set_ylabel(ylabel,   fontsize=11)
        ax.set_title(ylabel,    fontsize=12)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=9)

    # legenda aggiuntiva per il pannello frazioni
    axes[3].plot([], [], 'k-',  lw=2,   label='AEI (tutti e tre)')
    axes[3].plot([], [], 'k--', lw=1, alpha=0.5, label='k fisico')
    axes[3].plot([], [], 'k:',  lw=1, alpha=0.5, label='β ≤ 1')
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  HEATMAP  frac_aei  nello spazio dei parametri, per bin radiale
# ═══════════════════════════════════════════════════════════════════════════════

def plot_validity_heatmap(df_all, param_x='B00', param_y='Sigma0',
                          r_range=(None, None), zone=None,
                          n_bins_x=12, n_bins_y=12, figsize=(10, 7)):
    """
    Heatmap 2D: asse x = param_x, asse y = param_y,
    colore = frazione di punti AEI validi nel range radiale [r_min, r_max].

    Parameters
    ----------
    df_all   : DataFrame da radial_scan_grid
    param_x  : str  asse x ('a','B00','Sigma0')
    param_y  : str  asse y ('a','B00','Sigma0')
    r_range  : (r_min, r_max) | (None, None) → tutto il range
    zone     : str | None  — se specificato filtra per zona ('A','B','C')
    """
    sub = df_all.copy()
    if zone:
        sub = sub[sub['zone'] == zone]
    r_lo = r_range[0] or sub['r'].min()
    r_hi = r_range[1] or sub['r'].max()
    sub  = sub[(sub['r'] >= r_lo) & (sub['r'] <= r_hi)]

    if sub.empty:
        print("Nessun dato nel range selezionato.")
        return

    # bin 2D
    log_x = param_x in ('B00', 'Sigma0')
    log_y = param_y in ('B00', 'Sigma0')

    x_vals = np.log10(sub[param_x]) if log_x else sub[param_x]
    y_vals = np.log10(sub[param_y]) if log_y else sub[param_y]

    x_edges = np.linspace(x_vals.min(), x_vals.max(), n_bins_x + 1)
    y_edges = np.linspace(y_vals.min(), y_vals.max(), n_bins_y + 1)

    grid_frac = np.full((n_bins_y, n_bins_x), np.nan)
    grid_n    = np.zeros((n_bins_y, n_bins_x), dtype=int)

    for ix in range(n_bins_x):
        for iy in range(n_bins_y):
            mask = ((x_vals >= x_edges[ix]) & (x_vals < x_edges[ix+1]) &
                    (y_vals >= y_edges[iy]) & (y_vals < y_edges[iy+1]))
            pts = sub[mask]
            if len(pts) >= 3:
                grid_frac[iy, ix] = pts['aei_valid'].mean()
                grid_n[iy, ix]    = len(pts)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid_frac, origin='lower', aspect='auto',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   vmin=0, vmax=1, cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Frazione AEI valida')

    xlabel = f"log₁₀({param_x})" if log_x else param_x
    ylabel = f"log₁₀({param_y})" if log_y else param_y
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    zone_str = f"Zona {zone}" if zone else "Tutte le zone"
    ax.set_title(
        f"Frazione AEI valida — {zone_str}\n"
        f"r ∈ [{r_lo:.1f}, {r_hi:.1f}] rg", fontsize=12
    )
    ax.grid(False)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DISTRIBUZIONI DEGLI ESPONENTI  (power-law slopes vs parametri)
# ═══════════════════════════════════════════════════════════════════════════════

def _pl_slope(x, y):
    """Fit log-log lineare, restituisce pendenza o NaN."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan
    lx = np.log10(x[mask]); ly = np.log10(y[mask])
    mx, my = lx.mean(), ly.mean()
    denom = ((lx - mx)**2).sum()
    if denom == 0:
        return np.nan
    return float(((lx - mx) * (ly - my)).sum() / denom)


def compute_slopes_grid(df_all):
    """
    Per ogni run (a, B00, Sigma0) e ogni zona calcola le pendenze
    delle leggi di potenza B∝r^α, Σ∝r^α, β∝r^α, k·r∝r^α.

    Returns
    -------
    df_slopes : DataFrame con colonne:
        a, B00, Sigma0, zone, slope_B, slope_S, slope_beta, slope_kr
    """
    records = []
    for (a_val, B00_val, S0_val), grp in df_all.groupby(['a','B00','Sigma0']):
        for zone in ZONE_NAMES:
            sub = grp[grp['zone'] == zone]
            if len(sub) < 5:
                continue
            r = sub['r'].values
            records.append({
                'a': a_val, 'B00': B00_val, 'Sigma0': S0_val, 'zone': zone,
                'slope_B':    _pl_slope(r, sub['B0'].values),
                'slope_S':    _pl_slope(r, sub['Sigma'].values),
                'slope_beta': _pl_slope(r, sub['beta'].values),
                'slope_kr':   _pl_slope(r, sub['kr'].values),
            })
    return pd.DataFrame(records)


def plot_slope_distributions(df_slopes, figsize=(16, 10)):
    """
    Violinplot delle distribuzioni degli esponenti per zona,
    con overlay scatter colorato per spin.
    """
    quantities = [
        ('slope_B',    'B ∝ r^α   →   α'),
        ('slope_S',    'Σ ∝ r^α   →   α'),
        ('slope_beta', 'β ∝ r^α   →   α'),
        ('slope_kr',   'k·r ∝ r^α →   α'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=False)
    fig.suptitle("Distribuzioni degli esponenti delle leggi di potenza", fontsize=13)

    for ax, (col, label) in zip(axes, quantities):
        data_by_zone = [
            df_slopes[df_slopes['zone'] == z][col].dropna().values
            for z in ZONE_NAMES
        ]
        # violinplot
        parts = ax.violinplot(
            [d for d in data_by_zone if len(d) > 1],
            positions=range(len(ZONE_NAMES)),
            showmedians=True, showextrema=True
        )
        for pc, zone in zip(parts['bodies'], ZONE_NAMES):
            pc.set_facecolor(ZONE_COLORS[zone])
            pc.set_alpha(0.5)

        # scatter sovrapposto colorato per spin
        for i, zone in enumerate(ZONE_NAMES):
            sub = df_slopes[df_slopes['zone'] == zone].dropna(subset=[col])
            if sub.empty:
                continue
            sc = ax.scatter(
                np.random.normal(i, 0.05, len(sub)),
                sub[col],
                c=sub['a'], cmap='RdBu', vmin=-1, vmax=1,
                s=12, alpha=0.6, zorder=3
            )

        ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.set_xticks(range(len(ZONE_NAMES)))
        ax.set_xticklabels([f'Zona {z}' for z in ZONE_NAMES])
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label.split('→')[0].strip(), fontsize=11)
        ax.grid(True, alpha=0.15, axis='y')

    # colorbar spin
    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=Normalize(-1, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[-1], label='Spin a', shrink=0.8)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TABELLA RIASSUNTIVA GRIGLIA
# ═══════════════════════════════════════════════════════════════════════════════

def summary_table_grid(df_all, df_binned, df_slopes=None):
    """Stampa statistiche aggregate per zona sulla griglia completa."""
    print("\n" + "="*70)
    print("  ANALISI RADIALE — SINTESI GRIGLIA")
    print("="*70)

    n_runs = df_all.groupby(['a','B00','Sigma0']).ngroups
    print(f"  Run totali: {n_runs}   Punti totali: {len(df_all)}")
    print(f"  Range r: [{df_all['r'].min():.1f}, {df_all['r'].max():.0f}] rg\n")

    header = (f"{'Zona':>5} {'N punti':>9} {'% k ok':>8} {'% β≤1':>8} "
              f"{'% shear':>8} {'% AEI':>8}")
    if df_slopes is not None:
        header += f"  {'<slope_B>':>10} {'<slope_Σ>':>10} {'<slope_β>':>10}"
    print(header)
    print("-"*70)

    for zone in ZONE_NAMES:
        sub = df_all[df_all['zone'] == zone]
        N   = len(sub)
        if N == 0:
            continue
        pct = lambda c: f"{sub[c].sum()/N*100:.0f}%"
        row = (f"{zone:>5} {N:>9} {pct('k_valid'):>8} {pct('beta_valid'):>8} "
               f"{pct('shear_valid'):>8} {pct('aei_valid'):>8}")
        if df_slopes is not None:
            sz = df_slopes[df_slopes['zone'] == zone]
            for sc in ['slope_B','slope_S','slope_beta']:
                v = sz[sc].median() if not sz.empty else np.nan
                row += f"  {v:>10.3f}"
        print(row)

    print("="*70)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CELLE PRONTE PER IL NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════════════
"""
# ── Cella A: import ──────────────────────────────────────────────────────────

from radial_grid_analysis import (
    radial_scan_grid,
    plot_radial_grid,
    plot_validity_heatmap,
    compute_slopes_grid,
    plot_slope_distributions,
    summary_table_grid,
)

# ── Cella B: scan su griglia ─────────────────────────────────────────────────

params_grid = {
    'a':      (-0.9, 0.9, 7),
    'B00':    (1e4,  1e8, 8),
    'Sigma0': (1e3,  1e7, 8),
}

df_all, df_binned, meta_list = radial_scan_grid(
    params_grid,
    n_r=150,          # punti radiali per profilo
    n_rbins=30,       # bin radiali per aggregazione
    verbose=True,
)

# ── Cella C: profili radiali mediana ± IQR ───────────────────────────────────

fig = plot_radial_grid(df_binned, title="Analisi radiale — griglia parametri")
plt.show()

# ── Cella D: heatmap nello spazio dei parametri ──────────────────────────────

# per zona A, r < 20 rg
fig = plot_validity_heatmap(df_all, param_x='B00', param_y='Sigma0',
                            r_range=(None, 20), zone='A')
plt.show()

# per zona B, r 20–100 rg
fig = plot_validity_heatmap(df_all, param_x='a', param_y='B00',
                            r_range=(20, 100), zone='B')
plt.show()

# ── Cella E: distribuzioni degli esponenti ───────────────────────────────────

df_slopes = compute_slopes_grid(df_all)
fig = plot_slope_distributions(df_slopes)
plt.show()

# ── Cella F: tabella riassuntiva ─────────────────────────────────────────────

summary_table_grid(df_all, df_binned, df_slopes)
"""
