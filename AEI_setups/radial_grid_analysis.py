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
    n_r=500,
    r_max=1e6,
    r_max_aei=1000.0,
    n_points_ext=80,
    quantities=('k', 'beta', 'dQdr'),
    n_rbins=30,
    M=M_BH,
    verbose=True,
):
    """
    Itera su una griglia di parametri e per ogni combinazione calcola
    il profilo radiale completo tramite compute_disk_profile, poi aggrega
    i risultati in bin radiali log-spaziati.
 
    Funziona con qualsiasi griglia parametrica:
      • Simple:  {'a': array, 'B00': array, 'Sigma0': array}
      • SS/NT:   {'a': array, 'mdot': array}
      • generica: qualsiasi dict {nome: array_1d}
 
    Parameters
    ----------
    param_dict : dict {nome: array_1d}
        Ogni chiave è un nome di parametro, ogni valore è un array 1D.
        Il prodotto cartesiano viene calcolato internamente.
    disk_model : callable
        disk_model(r_rg, **params) → (B0, Sigma, c_s, hr, zone, info)
    mm : int
        Modo azimutale m della perturbazione AEI.
    hr : float
        Aspect ratio H/r (fallback se non restituito dal modello).
    n_r : int
        Numero di punti radiali per ogni profilo individuale.
    r_max : float o None
        Raggio esterno [r_g]. Se None: ricavato da info['r_BC'] se disponibile.
    quantities : tuple of str
        Colonne del DataFrame da aggregare nei bin radiali.
    n_rbins : int
        Numero di bin radiali log-spaziati.
    M : float
        Massa del buco nero [M_sun].
    verbose : bool
        Stampa progressi e statistiche.
 
    Returns
    -------
    df_all : pd.DataFrame
        Tutti i punti radiali di tutti i run concatenati.
        Contiene le colonne fisiche standard + una colonna per ogni chiave
        di param_dict.
    df_binned : pd.DataFrame
        Statistiche per bin radiale: r_mid, zone, {qty}_median/q1/q3/mean/std,
        count, frac_k, frac_beta, frac_shear, frac_aei.
    meta_list : list of dict
        Metadati (r_AB, r_BC, r_ISCO, …) per ogni run.
    """
    # ── prodotto cartesiano dei parametri ────────────────────────────────────
    param_keys = list(param_dict.keys())
    param_arrs = [np.asarray(param_dict[k]) for k in param_keys]
    grids      = np.meshgrid(*param_arrs, indexing='ij')
    flat       = {k: g.ravel() for k, g in zip(param_keys, grids)}
    total      = flat[param_keys[0]].size
 
    if verbose:
        dims = ' × '.join(str(len(a)) for a in param_arrs)
        print(f"Grid scan: {dims} = {total} combinazioni")
        print(f"Parametri: {param_keys}")
 
    all_frames = []
    meta_list  = []
    done = 0
 
    for i in range(total):
        params = {k: float(flat[k][i]) for k in param_keys}
        try:
            df_run, meta = compute_disk_profile(
                disk_model   = disk_model,
                params       = params,
                mm           = mm,
                hr           = hr,
                M            = M,
                n_points     = n_r,
                r_max        = r_max,
                r_max_aei    = r_max_aei,
                n_points_ext = n_points_ext,
            )
            for k, v in params.items():
                df_run[k] = v
            all_frames.append(df_run)
            meta_list.append(meta)
 
        except Exception as e:
            if verbose:
                param_str = '  '.join(f'{k}={v:.3g}' for k, v in params.items())
                print(f"  skip {param_str}: {e}")
 
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
    """Stampa veloce della frazione AEI per zona, incluse le colonne ILR se presenti."""
    has_ilr = 'ilr_valid' in df_all.columns
    header = "  Zona  |  N punti  | % k ok | % β≤1  | % shear | % AEI"
    if has_ilr:
        header += " | % r<ILR | % QPO"
    print("\n" + header)
    print("  " + "-" * (len(header) - 2))
    for zone in df_all['zone'].unique():
        sub = df_all[df_all['zone'] == zone]
        N = len(sub)
        if N == 0:
            continue
        pk = sub['k_valid'].sum()    / N * 100
        pb = sub['beta_valid'].sum() / N * 100
        ps = sub['shear_valid'].sum()/ N * 100
        pa = sub['aei_valid'].sum()  / N * 100
        line = f"    {zone}   | {N:>9} | {pk:>5.1f}% | {pb:>5.1f}% | {ps:>6.1f}% | {pa:>5.1f}%"
        if has_ilr:
            pi = sub['ilr_valid'].sum()     / N * 100
            pq = sub['aei_ilr_valid'].sum() / N * 100
            line += f" | {pi:>6.1f}% | {pq:>5.1f}%"
        print(line)
        

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
        'k':     'k  (dimensionless)',
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
            label = f'Zona {zone}' if zone != 'N/A' else 'disc'
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

def plot_validity_heatmap(df_all, param_x='a', param_y='mdot',
                          r_range=(None, None), zone=None,
                          metric='aei_valid',
                          n_bins_x=15, n_bins_y=15, figsize=(10, 7)):
    """
    Heatmap 2D: asse x = param_x, asse y = param_y,
    colore = metrica aggregata nel range radiale e zona selezionati.

    Funziona con qualsiasi coppia di parametri presenti in df_all.
    La scala logaritmica è applicata automaticamente per i parametri
    che spaziano più di 2 decadi e sono tutti positivi.

    Parameters
    ----------
    df_all   : DataFrame da radial_scan_grid
    param_x  : str  colonna asse x
    param_y  : str  colonna asse y
    r_range  : (r_min, r_max) | (None, None)
    zone     : str | None
    metric   : str  — colonna booleana da aggregare come media
    n_bins_x, n_bins_y : int
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

    def _should_log(col):
        vals = sub[col].values.astype(float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) == 0:
            return False
        return np.log10(vals.max() / vals.min()) > 2

    log_x = _should_log(param_x)
    log_y = _should_log(param_y)

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

    _known_labels = {
        'B00':    'B₀₀ [G]',
        'Sigma0': 'Σ₀ [g/cm²]',
        'mdot':   'ṁ = Ṁ/Ṁ_Edd',
        'a':      'spin  a',
    }
    def _axis_label(p, is_log):
        name = _known_labels.get(p, p)
        return f'log₁₀({name})' if is_log else name

    ax.set_xlabel(_axis_label(param_x, log_x), fontsize=12)
    ax.set_ylabel(_axis_label(param_y, log_y), fontsize=12)

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
    Per ogni run e ogni zona calcola le pendenze delle leggi di potenza
    B∝r^α, Σ∝r^α, β∝r^α.

    Funziona con qualsiasi griglia parametrica (Simple o SS/NT): le chiavi
    di raggruppamento vengono ricavate automaticamente come le colonne
    non-fisiche presenti in df_all.

    Parameters
    ----------
    df_all : DataFrame da radial_scan_grid

    Returns
    -------
    df_slopes : DataFrame
        Colonne: tutte le chiavi di gruppo (es. a/B00/Sigma0 o a/mdot),
                 zone, slope_B, slope_S, slope_beta.
    """
    _physics_cols = {'r', 'zone', 'B0', 'Sigma', 'c_s', 'hr', 'k', 'beta',
                     'dQdr', 'k_valid', 'beta_valid', 'shear_valid', 'aei_valid',
                     'ilr_valid', 'aei_ilr_valid', 'r_bin'}
    group_keys = [c for c in df_all.columns if c not in _physics_cols]

    records = []
    zones_present = df_all['zone'].unique().tolist()
    for combo, grp in df_all.groupby(group_keys):
        if not isinstance(combo, tuple):
            combo = (combo,)
        combo_dict = dict(zip(group_keys, combo))
        for zone in zones_present:
            sub = grp[grp['zone'] == zone]
            if len(sub) < 5:
                continue
            r = sub['r'].values
            rec = {
                **combo_dict,
                'zone':       zone,
                'slope_B':    _pl_slope(r, sub['B0'].values)    if 'B0'    in sub else np.nan,
                'slope_S':    _pl_slope(r, sub['Sigma'].values) if 'Sigma' in sub else np.nan,
                'slope_beta': _pl_slope(r, sub['beta'].values)  if 'beta'  in sub else np.nan,
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

    Funziona con qualsiasi griglia parametrica (Simple o SS/NT): il numero
    di run viene contato raggruppando per le colonne parametriche effettive.

    Parameters
    ----------
    df_all     : DataFrame da radial_scan_grid
    df_binned  : DataFrame (opzionale, per info aggiuntive sui bin)
    df_slopes  : DataFrame da compute_slopes_grid (opzionale)

    Returns
    -------
    df_summary : pd.DataFrame  con le stesse info stampate
    """
    _physics_cols = {'r', 'zone', 'B0', 'Sigma', 'c_s', 'hr', 'k', 'beta',
                     'dQdr', 'k_valid', 'beta_valid', 'shear_valid', 'aei_valid',
                     'ilr_valid', 'aei_ilr_valid', 'r_bin'}
    group_keys = [c for c in df_all.columns if c not in _physics_cols]
    n_runs = df_all.groupby(group_keys).ngroups
    print("\n" + "=" * 78)
    print("  ANALISI RADIALE — SINTESI GRIGLIA")
    print("=" * 78)
    print(f"  Run totali: {n_runs}   |   Punti totali: {len(df_all)}")
    print(f"  Range r:    [{df_all['r'].min():.1f}, {df_all['r'].max():.0f}] rg")
    if df_binned is not None:
        print(f"  Bin radiali con dati: {len(df_binned)}")
    print()

    has_slopes = df_slopes is not None
    has_ilr    = 'ilr_valid' in df_all.columns

    header = (f"{'Zona':>5}  {'N':>9}  {'% k':>6}  {'% β':>6}  "
              f"{'% sh':>6}  {'% AEI':>7}")
    if has_ilr:
        header += f"  {'% ILR':>7}  {'% QPO':>7}"
    if has_slopes:
        header += f"  {'α_B':>8}  {'α_Σ':>8}  {'α_β':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    zones_iter = df_all['zone'].unique().tolist()
    # prefer canonical order: A, B, C first, then N/A and others
    _order = ['A', 'B', 'C']
    zones_iter = [z for z in _order if z in zones_iter] + \
                 [z for z in zones_iter if z not in _order]
    for zone in zones_iter:
        sub = df_all[df_all['zone'] == zone]
        N = len(sub)
        if N == 0:
            continue
        pct = lambda c: sub[c].sum() / N * 100 if c in sub.columns else 0.0

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
                f"{pct('aei_valid'):>6.3f}%")

        if has_ilr:
            p_ilr = pct('ilr_valid')
            p_qpo = pct('aei_ilr_valid')
            row_dict['pct_ilr'] = p_ilr
            row_dict['pct_qpo'] = p_qpo
            line += f"  {p_ilr:>6.1f}%  {p_qpo:>6.3f}%"

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

def compare_models(results_dict, figsize=(16, 10),
                   aei_only=False,
                   show_iqr=True,
                   a_ref=None, nu_obs=None, m_mode=None, M_bh=None):
    """
    Confronta i profili radiali aggregati di B₀, Σ, k, β per più modelli,
    con lo stesso stile di plot_profiles_comparison ma calcolati come
    mediana (± IQR opzionale) sulla griglia di parametri.

    Parameters
    ----------
    results_dict : dict  { 'nome_modello': (df_all, df_binned, meta_list) }
    figsize      : tuple
    aei_only     : bool (default False)
        True  → aggrega solo i punti con aei_valid=True
        False → aggrega tutti i punti della griglia
    show_iqr     : bool (default True)
        True  → banda IQR (Q1–Q3) attorno alla mediana
    a_ref        : float | None
        Spin per calcolare ILR/OLR/CR. Default: mediana del primo modello.
    nu_obs, m_mode, M_bh : override dei default di aei_common

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # ── importa funzioni di risonanza ────────────────────────────────────────
    try:
        from .aei_common import (r_ilr as _r_ilr, r_olr as _r_olr,
                                  r_corotation as _r_cr,
                                  NU0 as _NU0, mm as _mm, M_BH as _M_BH)
    except ImportError:
        from aei_common import (r_ilr as _r_ilr, r_olr as _r_olr,
                                 r_corotation as _r_cr,
                                 NU0 as _NU0, mm as _mm, M_BH as _M_BH)

    nu_obs = nu_obs if nu_obs is not None else _NU0
    m_mode = m_mode if m_mode is not None else _mm
    M_bh   = M_bh   if M_bh   is not None else _M_BH

    first_df_all = next(iter(results_dict.values()))[0]
    if a_ref is None:
        a_ref = float(np.median(first_df_all['a'].unique()))

    r_ILR = _r_ilr(a_ref, nu_obs, m_mode, M_bh)
    r_CR  = _r_cr (a_ref, nu_obs, m_mode, M_bh)
    r_OLR = _r_olr(a_ref, nu_obs, m_mode, M_bh)
    resonances = [
        (r_ILR, r'$r_{\rm ILR}$', 'blue',    '--'),
        (r_OLR, r'$r_{\rm OLR}$', 'magenta', '--'),
        (r_CR,  r'$r_{\rm CR}$',  '#f43f5e', '-.'),
    ]

    # ── panels: same 4 quantities as plot_profiles_comparison ────────────────
    panels = [
        # (col_in_df, col_in_binned, ylabel, log_y, hlines)
        ('B0',    'B0',    r'B₀  [G]',              True,  []),
        ('Sigma', 'Sigma', r'Σ  [g/cm²]',           True,  []),
        ('k',     'k',     r'k  (dimensionless)',    True,  [(0.1,'gray',':'),(10,'gray',':')]),
        ('beta',  'beta',  r'β',                     True,  [(1.0,'red','--')]),
    ]

    import matplotlib.gridspec as _gs
    fig = plt.figure(figsize=figsize)
    mode_label = 'AEI-valid only' if aei_only else 'full grid'
    fig.suptitle(
        f"Models comparison — {mode_label} — "
        f"median ± IQR  (a_ref = {a_ref:.2f})",
        fontsize=13
    )
    gs = _gs.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    linestyles   = ['-', '--', ':', '-.']
    cmap         = plt.cm.tab10
    model_colors = {name: cmap(i) for i, name in enumerate(results_dict)}

    for ax, (col_df, col_bin, ylabel, log_y, hrefs) in zip(axes, panels):

        for (name, (df_all, df_binned, _)), ls in zip(results_dict.items(), linestyles):
            col = model_colors[name]

            # ── source data ──────────────────────────────────────────────────
            df_src = df_all[df_all['aei_valid']] if aei_only else df_all

            if df_src.empty:
                continue

            # Re-bin to get median ± IQR per r_mid
            r_lo = max(df_src['r'].min(), 1.0)
            r_hi = df_src['r'].max()
            n_rbins = 40
            edges  = np.geomspace(r_lo, r_hi, n_rbins + 1)
            r_mids = np.sqrt(edges[:-1] * edges[1:])

            r_plot, med_vals, q1_vals, q3_vals = [], [], [], []
            for lo, hi, rm in zip(edges[:-1], edges[1:], r_mids):
                pts = df_src[(df_src['r'] >= lo) & (df_src['r'] < hi)][col_df].dropna()
                pts = pts[np.isfinite(pts) & (pts > 0)] if log_y else pts[np.isfinite(pts)]
                if len(pts) >= 3:
                    r_plot.append(rm)
                    med_vals.append(float(np.median(pts)))
                    q1_vals.append(float(np.percentile(pts, 25)))
                    q3_vals.append(float(np.percentile(pts, 75)))

            if not r_plot:
                continue

            r_arr  = np.array(r_plot)
            med    = np.array(med_vals)
            q1     = np.array(q1_vals)
            q3     = np.array(q3_vals)

            ax.plot(r_arr, med, color=col, ls=ls, lw=1.8,
                    label=name, alpha=0.9)
            if show_iqr:
                ax.fill_between(r_arr, q1, q3, color=col, alpha=0.15)

        # ── horizontal reference lines ────────────────────────────────────
        for (yv, hc, hls) in hrefs:
            ax.axhline(yv, color=hc, ls=hls, lw=1, alpha=0.7)

        # ── resonance vertical lines ──────────────────────────────────────
        for (rr, rlabel, rc, rls) in resonances:
            if np.isfinite(rr):
                ax.axvline(rr, color=rc, ls=rls, lw=1.3, alpha=0.85,
                           label=f'{rlabel} = {rr:.1f}')

        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel('r  [r_g]', fontsize=11)
        ax.set_ylabel(ylabel,     fontsize=11)
        ax.set_title(ylabel,      fontsize=11)
        ax.grid(True, alpha=0.15)

        # legend: models first, then resonances
        handles_m = [plt.Line2D([0],[0], color=model_colors[n], ls=l, lw=2, label=n)
                     for n, l in zip(results_dict, linestyles)]
        handles_r = [plt.Line2D([0],[0], color=rc, ls=rls, lw=1.3,
                                 label=f'{rlabel} = {rr:.1f}')
                     for (rr, rlabel, rc, rls) in resonances if np.isfinite(rr)]
        ax.legend(handles=handles_m + handles_r, fontsize=8)

    plt.tight_layout()
    return fig