"""
plot_disk_profiles.py
=====================
Funzioni di plotting universali per l'analisi AEI.

Compatibili con tutti i modelli di disco (simple_disc, full_disk_SS, NT)
tramite l'output standard di aei_common.compute_disk_profile:

    df   : pd.DataFrame  con colonne r, zone, B0, Sigma, c_s, k,
                         beta, dQdr, k_valid, beta_valid, shear_valid, aei_valid
    meta : dict          con almeno r_H, r_ISCO, a, B00, Sigma0, mm, hr, M
                         + opzionali: r_AB, r_BC, mdot, alpha

Funzioni principali
-------------------
  plot_full_disk_profiles    4 pannelli: B/Σ, k·r, β, dQ/dr  con zone colorate
  plot_aei_validity_map      4 barre orizzontali con i criteri AEI per ogni r
  plot_summary_table         tabella riassuntiva per zona (stampata + ritornata)

Tutte le funzioni gestiscono trasparentemente:
  • presenza o assenza di zone (colonna 'zone' con valori 'A','B','C' o 'N/A')
  • presenza o assenza di frontiere r_AB, r_BC nel meta
  • presenza o assenza di mdot, alpha nel meta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

# ── costanti di layout per le zone ──────────────────────────────────────────
ZONE_NAMES  = ['A', 'B', 'C']
ZONE_COLORS = {'A': '#f97316', 'B': '#3b82f6', 'C': '#22c55e', 'N/A': '#94a3b8'}

# colori delle linee di frontiera
_COL_ISCO = '#ffffff'
_COL_AB   = '#f97316'
_COL_BC   = '#3b82f6'
_COL_ILR  = '#facc15'   # giallo — Inner Lindblad Resonance (cavity AEI)
_COL_CR   = '#f43f5e'   # rosa — Corotation Radius


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNI
# ═══════════════════════════════════════════════════════════════════════════════

def _has_zones(df):
    """True se il DataFrame contiene zone vere (non solo 'N/A')."""
    return 'zone' in df.columns and set(df['zone'].unique()) != {'N/A'}


def _zone_vlines(ax, meta, alpha=0.7):
    """
    Aggiunge linee verticali per r_ISCO (sempre), r_AB e r_BC (se nel meta).
    Non lancia eccezioni se le frontiere mancano.
    """
    if 'r_ISCO' in meta:
        ax.axvline(meta['r_ISCO'], color=_COL_ISCO, ls=':', lw=1, alpha=0.45,
                   label=f"r_ISCO={meta['r_ISCO']:.1f}")
    if 'r_AB' in meta:
        ax.axvline(meta['r_AB'], color=_COL_AB, ls='--', lw=1, alpha=alpha,
                   label=f"r_AB={meta['r_AB']:.1f}")
    if 'r_BC' in meta:
        ax.axvline(meta['r_BC'], color=_COL_BC, ls='--', lw=1, alpha=alpha,
                   label=f"r_BC={meta['r_BC']:.0f}")


def _build_title(meta, alpha_visc=None):
    """Costruisce il titolo della figura dai meta disponibili."""
    parts = [f"a={meta.get('a', '?'):.2f}"]
    if 'B00' in meta:
        parts.append(f"B₀₀={meta['B00']:.1e}")
    if 'Sigma0' in meta:
        parts.append(f"Σ₀={meta['Sigma0']:.1e}")
    if 'mdot' in meta:
        parts.append(f"ṁ={meta['mdot']:.3f}")
    alpha = meta.get('alpha', alpha_visc)
    if alpha is not None:
        parts.append(f"α={alpha:.2f}")
    return "  |  ".join(parts)


def _iter_zones(df):
    """
    Genera (zone_label, subset_df, color) per ogni gruppo presente nel DataFrame.
    Se le zone sono tutte 'N/A' produce un solo gruppo con il DataFrame intero.
    """
    if _has_zones(df):
        for zn in ZONE_NAMES:
            sub = df[df['zone'] == zn]
            if not sub.empty:
                yield zn, sub, ZONE_COLORS[zn]
    else:
        yield 'disco', df, ZONE_COLORS['N/A']


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PLOT PROFILI COMPLETI
# ═══════════════════════════════════════════════════════════════════════════════

def plot_full_disk_profiles(df, meta, alpha_visc=None, figsize=(18, 13)):
    """
    Figura con 6 pannelli: B₀, Σ, H/r, k, β, dQ/dr.

    Funziona con qualsiasi output di compute_disk_profile:
      • modelli con zone A/B/C (SS, NT): ogni curva è colorata per zona
      • modelli senza zone (simple_disc):  un'unica curva grigia

    Il pannello H/r (aspect ratio) mostra il valore fisico calcolato dal
    modello (SS/NT zona-per-zona) o il valore costante HOR (Simple).
    La colonna 'hr' nel DataFrame viene prodotta da compute_disk_profile
    come quarto elemento del return di disk_model.

    Parameters
    ----------
    df         : DataFrame da compute_disk_profile
    meta       : dict dei metadati (r_ISCO, r_AB, r_BC se disponibili, ecc.)
    alpha_visc : float opzionale — usato solo se meta non contiene 'alpha'
    figsize    : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(_build_title(meta, alpha_visc), fontsize=13)
    gs   = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.34)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    aei = df[df['aei_valid']]

    # ── pannello 0: B₀ ────────────────────────────────────────────────────────
    ax = axes[0]
    for label, sub, col in _iter_zones(df):
        valid_B = sub[sub['B0'] > 0]
        if not valid_B.empty:
            ax.semilogy(valid_B['r'], valid_B['B0'],
                        color=col, lw=2, label=f"Zona {label}")
    ax.set_xscale('log')
    ax.set_xlabel('r [rg]')
    ax.set_ylabel('B₀  [G]')
    ax.set_title('Magnetic field  B₀(r)')
    _zone_vlines(ax, meta)
    ax.legend(fontsize=8, loc='upper right')

    # ── pannello 1: Σ ─────────────────────────────────────────────────────────
    ax = axes[1]
    for label, sub, col in _iter_zones(df):
        valid_S = sub[sub['Sigma'] > 0]
        if not valid_S.empty:
            ax.semilogy(valid_S['r'], valid_S['Sigma'],
                        color=col, lw=2, label=f"Zona {label}")
    ax.set_xscale('log')
    ax.set_xlabel('r [rg]')
    ax.set_ylabel('Σ  [g/cm²]')
    ax.set_title('Surface density  Σ(r)')
    _zone_vlines(ax, meta)
    ax.legend(fontsize=8, loc='upper right')

    # ── pannello 2: H/r ───────────────────────────────────────────────────────
    ax = axes[2]
    has_hr = 'hr' in df.columns and df['hr'].notna().any() and (df['hr'] > 0).any()
    if has_hr:
        for label, sub, col in _iter_zones(df):
            valid_hr = sub[sub['hr'] > 0]
            if not valid_hr.empty:
                ax.semilogy(valid_hr['r'], valid_hr['hr'],
                            color=col, lw=2, label=f"Zona {label}")
        # se il valore è costante (Simple), evidenziarlo
        hr_std = df['hr'].std()
        if hr_std < 1e-10 * df['hr'].mean():
            hr_val = df['hr'].mean()
            ax.axhline(hr_val, color='white', ls=':', lw=1.2, alpha=0.5,
                       label=f'H/r = {hr_val:.3f} (costante)')
    else:
        # fallback: usa hr da meta se disponibile
        hr_val = meta.get('hr', None)
        if hr_val is not None:
            ax.axhline(hr_val, color='#94a3b8', ls='--', lw=2,
                       label=f'H/r = {hr_val:.3f} (meta)')
        ax.text(0.5, 0.5, 'hr non disponibile', ha='center', va='center',
                transform=ax.transAxes, color='gray', fontsize=10)
    ax.set_xscale('log')
    ax.set_xlabel('r [rg]')
    ax.set_ylabel('H/r')
    ax.set_title('Aspect ratio  H/r(r)')
    _zone_vlines(ax, meta)
    ax.legend(fontsize=8)

    # ── pannello 3: k ─────────────────────────────────────────────────────────
    ax = axes[3]
    for label, sub, col in _iter_zones(df):
        valid = sub[sub['k'].notna() & (sub['k'] > 0)]
        if not valid.empty:
            ax.semilogy(valid['r'], valid['k'],
                        color=col, lw=2, label=f"Zona {label}")
    ax.axhline(0.1, color='gray', ls=':', lw=1, alpha=0.7)
    ax.axhline(10,  color='gray', ls=':', lw=1, alpha=0.7, label='limiti WKB')
    if not aei.empty:
        ax.scatter(aei['r'], aei['k'],
                   color='yellow', s=8, zorder=5, alpha=0.5, label='AEI valida')
    ax.set_xscale('log')
    ax.set_xlabel('r [rg]')
    ax.set_ylabel('k  (dimensionless)')
    ax.set_title('Wavenumber  k(r)')
    _zone_vlines(ax, meta)
    ax.legend(fontsize=8)

    # ── pannello 4: β ─────────────────────────────────────────────────────────
    ax = axes[4]
    for label, sub, col in _iter_zones(df):
        valid = sub[sub['beta'] > 0]
        if not valid.empty:
            ax.semilogy(valid['r'], valid['beta'],
                        color=col, lw=2, label=f"Zona {label}")
    ax.axhline(1.0, color='red', ls='--', lw=1.2, label='β = 1')
    ax.set_xscale('log')
    ax.set_xlabel('r [rg]')
    ax.set_ylabel('β')
    ax.set_title('Magnetisation β(r)')
    _zone_vlines(ax, meta)
    ax.legend(fontsize=8)

    # ── pannello 5: dQ/dr ─────────────────────────────────────────────────────
    ax = axes[5]
    for label, sub, col in _iter_zones(df):
        ax.plot(sub['r'], sub['dQdr'], color=col, lw=2, label=f"Zona {label}")
    ax.axhline(0, color='red', ls='--', lw=1.2, label='dQ/dr = 0')
    if not aei.empty:
        ax.fill_between(df['r'], df['dQdr'],
                        where=df['aei_valid'],
                        color='yellow', alpha=0.15, label='AEI valida')
    ax.set_xscale('log')
    ax.set_xlabel('r [rg]')
    ax.set_ylabel('dQ/dr  [u.a.]')
    ax.set_title('Shear condition  dQ/dr(r)')
    _zone_vlines(ax, meta)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MAPPA DI VALIDITÀ AEI
# ═══════════════════════════════════════════════════════════════════════════════

def plot_aei_validity_map(df, meta, figsize=(14, 5)):
    """
    4 barre orizzontali (share x) che mostrano per ogni r
    quali criteri AEI sono soddisfatti.

    Funziona con e senza zone. Se le zone sono definite, la barra
    di sfondo è colorata per zona.

    Parameters
    ----------
    df     : DataFrame da compute_disk_profile
    meta   : dict dei metadati
    figsize: tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    criteria = [
        ('k_valid',     'WKB  (0.1 ≤ k·r ≤ 10)', '#a78bfa'),
        ('beta_valid',  'β ≤ 1',                   '#ef4444'),
        ('shear_valid', 'dQ/dr > 0',               '#f59e0b'),
        ('aei_valid',   'AEI  (WKB + β + shear)',  '#22c55e'),
    ]
    if 'ilr_valid' in df.columns:
        criteria += [
            ('ilr_valid',     'r < r_ILR  (cavity AEI)',       _COL_ILR),
            ('aei_ilr_valid', 'QPO candidate  (AEI + r<r_ILR)', '#10b981'),
        ]

    fig, axes = plt.subplots(len(criteria), 1, figsize=figsize, sharex=True)

    for ax, (col, label, color) in zip(axes, criteria):
        if col not in df.columns:
            ax.set_ylabel(label, fontsize=9, color='gray', labelpad=4)
            ax.set_yticks([])
            continue

        # sfondo per zona (se disponibile)
        if _has_zones(df):
            for zn, sub, zcol in _iter_zones(df):
                ax.axvspan(sub['r'].min(), sub['r'].max(),
                           alpha=0.07, color=zcol, zorder=0)

        ax.fill_between(df['r'], df[col].astype(float),
                        step='mid', color=color, alpha=0.75, zorder=2)
        ax.set_ylim(0, 1.35)
        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=9, color=color, labelpad=4)
        ax.grid(False)

        # linee frontiere SS/NT
        if 'r_AB' in meta:
            ax.axvline(meta['r_AB'], color=_COL_AB, ls='--', lw=0.9, alpha=0.65)
        if 'r_BC' in meta:
            ax.axvline(meta['r_BC'], color=_COL_BC, ls='--', lw=0.9, alpha=0.65)
        # linee risonanze AEI
        if 'r_ILR' in meta and np.isfinite(float(meta['r_ILR'])):
            ax.axvline(meta['r_ILR'], color=_COL_ILR, ls='-', lw=1.8, alpha=0.9)
        if 'r_CR' in meta and np.isfinite(float(meta['r_CR'])):
            ax.axvline(meta['r_CR'], color=_COL_CR, ls='--', lw=1.3, alpha=0.8)

    axes[-1].set_xscale('log')
    axes[-1].set_xlabel('r  [rg]', fontsize=11)

    fig.suptitle(
        f"AEI validity map  —  {_build_title(meta)}",
        fontsize=11
    )

    # legenda frontiere (su primo pannello)
    handles = []
    if 'r_ILR' in meta and np.isfinite(float(meta['r_ILR'])):
        handles.append(plt.Line2D([0], [0], color=_COL_ILR, ls='-', lw=1.8,
                                  label=f"r_ILR = {meta['r_ILR']:.1f} rg"))
    if 'r_CR' in meta and np.isfinite(float(meta['r_CR'])):
        handles.append(plt.Line2D([0], [0], color=_COL_CR, ls='--', lw=1.3,
                                  label=f"r_CR = {meta['r_CR']:.1f} rg"))
    if 'r_AB' in meta:
        handles.append(plt.Line2D([0], [0], color=_COL_AB, ls='--', lw=1,
                                  label=f"r_AB = {meta['r_AB']:.1f} rg"))
    if 'r_BC' in meta:
        handles.append(plt.Line2D([0], [0], color=_COL_BC, ls='--', lw=1,
                                  label=f"r_BC = {meta['r_BC']:.0f} rg"))
    if handles:
        axes[0].legend(handles=handles, fontsize=8, loc='upper right')

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TABELLA RIASSUNTIVA PER ZONA
# ═══════════════════════════════════════════════════════════════════════════════

def _pl_slope(r, y):
    """Fit log-log lineare → pendenza, o NaN se dati insufficienti."""
    mask = (r > 0) & (y > 0) & np.isfinite(r) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan
    lx = np.log10(r[mask])
    ly = np.log10(y[mask])
    mx, my = lx.mean(), ly.mean()
    denom = ((lx - mx) ** 2).sum()
    if denom == 0:
        return np.nan
    return float(((lx - mx) * (ly - my)).sum() / denom)


def plot_summary_table(df, meta, alpha_visc=None):
    """
    Stampa una tabella riassuntiva per zona con percentuali di validità
    e pendenze delle leggi di potenza B∝r^α, Σ∝r^α, β∝r^α.

    Funziona con e senza zone: se le zone sono 'N/A', riporta
    le statistiche sull'intero disco come riga unica.

    Parameters
    ----------
    df         : DataFrame da compute_disk_profile
    meta       : dict dei metadati
    alpha_visc : float opzionale (fallback se meta non ha 'alpha')

    Returns
    -------
    df_summary : pd.DataFrame  con le stesse informazioni in forma tabellare
    """
    alpha = meta.get('alpha', alpha_visc)
    mdot  = meta.get('mdot', float('nan'))

    print("\n" + "=" * 78)
    print(f"  SINTESI PROFILO DISCO  |  {_build_title(meta, alpha_visc)}")
    if not np.isnan(mdot):
        print(f"  ṁ = {mdot:.4f}   r_ISCO = {meta.get('r_ISCO', '?'):.2f} rg", end="")
    if 'r_AB' in meta:
        print(f"   r_AB = {meta['r_AB']:.1f} rg", end="")
    if 'r_BC' in meta:
        print(f"   r_BC = {meta['r_BC']:.0f} rg", end="")
    print()
    print("=" * 78)

    header = (f"{'Zona':>6}  {'N':>6}  {'% k':>6}  {'% β':>6}  "
              f"{'% sh':>6}  {'% AEI':>7}  {'α_B':>8}  {'α_Σ':>8}  {'α_β':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    # determina i gruppi da iterare
    if _has_zones(df):
        groups = [(zn, df[df['zone'] == zn]) for zn in ZONE_NAMES]
    else:
        groups = [('tutto', df)]

    records = []
    for zone_label, sub in groups:
        N = len(sub)
        if N == 0:
            continue

        pct = lambda c: sub[c].sum() / N * 100
        r   = sub['r'].values

        s_B    = _pl_slope(r, sub['B0'].values)    if 'B0'    in sub else np.nan
        s_S    = _pl_slope(r, sub['Sigma'].values) if 'Sigma' in sub else np.nan
        s_beta = _pl_slope(r, sub['beta'].values)  if 'beta'  in sub else np.nan

        def _fmt(v):
            return f"{v:+.3f}" if np.isfinite(v) else "  —  "

        line = (f"  {zone_label:>5}  {N:>6}  {pct('k_valid'):>5.1f}%  "
                f"{pct('beta_valid'):>5.1f}%  {pct('shear_valid'):>5.1f}%  "
                f"{pct('aei_valid'):>6.1f}%  "
                f"{_fmt(s_B):>8}  {_fmt(s_S):>8}  {_fmt(s_beta):>8}")
        print(line)

        records.append({
            'zone':       zone_label,
            'N':          N,
            'pct_k':      pct('k_valid'),
            'pct_beta':   pct('beta_valid'),
            'pct_shear':  pct('shear_valid'),
            'pct_aei':    pct('aei_valid'),
            'slope_B':    s_B,
            'slope_S':    s_S,
            'slope_beta': s_beta,
        })

    print("=" * 78)
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PLOT COMPARATIVO  (più modelli / più run sullo stesso asse)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_profiles_comparison(runs, figsize=(18, 13)):
    """
    Sovrappone i profili di più run (modelli o parametri diversi)
    sugli stessi 6 pannelli: B₀, Σ, H/r, k, β, dQ/dr.

    Il pannello H/r mostra il profilo fisico per SS/NT (zona-per-zona)
    oppure la costante HOR per i modelli Simple. La colonna 'hr' deve
    essere presente nel DataFrame (prodotta da compute_disk_profile).

    Parameters
    ----------
    runs : list of (label, df, meta)
        Ogni elemento è una tripla con:
          label : str   — nome del run (per la legenda)
          df    : DataFrame da compute_disk_profile
          meta  : dict dei metadati

    figsize : tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    cmap      = plt.cm.tab10
    run_cols  = {label: cmap(i) for i, (label, _, _) in enumerate(runs)}
    ls_cycle  = ['-', '--', ':', '-.', (0,(3,1,1,1))]

    fig = plt.figure(figsize=figsize)
    fig.suptitle("Disk profiles comparison", fontsize=13)
    gs   = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.34)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    panels = [
        # (qty_y, ylabel, logscale_y, hlines)
        ('B0',    'B₀  [G]',            True,  []),
        ('Sigma', 'Σ  [g/cm²]',         True,  []),
        ('hr',    'H/r',                True,  []),
        ('k',     'k  (dimensionless)', True,  [(0.1,'gray',':'), (10,'gray',':')]),
        ('beta',  'β',                  True,  [(1.0,'red','--')]),
        ('dQdr',  'dQ/dr  [u.a.]',      False, [(0.0,'red','--')]),
    ]

    for (label, df, meta), ls in zip(runs, ls_cycle):
        col = run_cols[label]

        for ax, (qty, ylabel, log_y, hrefs) in zip(axes, panels):
            if qty not in df.columns:
                continue
            valid = df[df[qty].notna()]
            if qty in ('B0', 'Sigma', 'beta', 'k', 'hr'):
                valid = valid[valid[qty] > 0]
            if not valid.empty:
                ax.plot(valid['r'], valid[qty],
                        color=col, ls=ls, lw=1.8, label=label, alpha=0.85)

    for ax, (qty, ylabel, log_y, hrefs) in zip(axes, panels):
        for (yval, hcol, hls) in hrefs:
            ax.axhline(yval, color=hcol, ls=hls, lw=1, alpha=0.7)
        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel('r [rg]', fontsize=11)
        ax.set_ylabel(ylabel,   fontsize=11)
        ax.set_title(ylabel,    fontsize=11)
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PLOT e stats
# ═══════════════════════════════════════════════════════════════════════════

def plot_standard_4panels(df, title_prefix=""):
    """
    Crea i 4 grafici standard per analizzare le soluzioni
    """
    if len(df) == 0:
        print(f"Nessuna soluzione per {title_prefix}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Panel 1: k vs r (colored by spin)
    sc1 = axes[0, 0].scatter(df['r'], df['k'], c=df['a'], alpha=0.6, cmap='RdBu', s=20)
    axes[0, 0].set_xlabel('r [r_g]', fontsize=12)
    axes[0, 0].set_ylabel('k [dimensionless]', fontsize=12)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title(f'{title_prefix}Wavenumber vs Radius', fontsize=13)
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axes[0, 0], label='Spin a')

    # Panel 2: B00 vs Sigma0 (colored by radius)
    sc2 = axes[0, 1].scatter(df['B00'], df['Sigma0'], c=df['r'], 
                             alpha=0.6, cmap='viridis', s=20)
    sc2.set_norm(LogNorm())
    axes[0, 1].set_xlabel('B₀₀', fontsize=12)
    axes[0, 1].set_ylabel('Σ₀', fontsize=12)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title(f'{title_prefix}Parameter Space', fontsize=13)
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=axes[0, 1], label='r [r_g]')

    # Panel 3: k (dimensionless) vs spin
    sc3 = axes[1, 0].scatter(df['a'], df['k'], c=df['r'], 
                             alpha=0.6, cmap='plasma', s=20)
    sc3.set_norm(LogNorm())
    axes[1, 0].set_xlabel('Spin a', fontsize=12)
    axes[1, 0].set_ylabel('k [dimensionless]', fontsize=12)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title(f'{title_prefix}Dimensionless wavenumber', fontsize=13)
    axes[1, 0].axhline(0.1, ls='--', c='gray', alpha=0.5, label='Physical range')
    axes[1, 0].axhline(10, ls='--', c='gray', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    plt.colorbar(sc3, ax=axes[1, 0], label='r [r_g]')
    
    # Panel 4: beta vs spin (colored by radius)
    sc4 = axes[1, 1].scatter(df['a'], df['beta'], c=df['r'], 
                             alpha=0.6, cmap='plasma', s=20)
    sc4.set_norm(LogNorm())
    axes[1, 1].set_xlabel('Spin a', fontsize=12)
    axes[1, 1].set_ylabel(r'$\beta$ [dimensionless]', fontsize=12)
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title(f'{title_prefix}Disk magnetization', fontsize=13)
    axes[1, 1].axhline(1, ls='--', c='red', alpha=0.7, label=r'$\beta = 1$')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    plt.colorbar(sc4, ax=axes[1, 1], label='r [r_g]')
    
    plt.tight_layout()
    plt.show()


def summarize_comparison2(dfs):
    """
    Confronta più DataFrame di soluzioni producendo:
    - tabella riassuntiva
    - grafico a barre
    
    Parametri
    ----------
    dfs : dict
        Dizionario {nome_Configuration: dataframe}
    """

    results = []

    for name, df in dfs.items():

        stats = {
            "Configuration": name,
            "# Solutions": len(df),
            "Range k":
                f"[{df['k'].min():.2e}, {df['k'].max():.2e}]",
            "Range β":
                f"[{df['beta'].min():.2e}, {df['beta'].max():.2e}]",
            "Range dQ/dr":
                f"[{df['dQdr'].min():.2e}, {df['dQdr'].max():.2e}]",
            "% β ≤ 1": (df['beta'] <= 1).mean() * 100,
            "% dQ/dr > 0": (df['dQdr'] > 0).mean() * 100
        }

        results.append(stats)

    comparison = pd.DataFrame(results)

    # baseline
    baseline = comparison.loc[
        comparison['Configuration'] == 'Baseline',
        '# Solutions'
    ].iloc[0]

    # salva valori numerici per il grafico
    solutions_numeric = comparison['# Solutions'].copy()

    # percentuali di rimanenti (solo array temporaneo)
    reductions = (solutions_numeric / baseline) * 100

    # formatta colonna # Solutions
    comparison['# Solutions'] = [
        f"{int(n)} ({r:.1f}%)" if cfg != "Baseline" else f"{int(n)}"
        for n, r, cfg in zip(solutions_numeric, reductions, comparison['Configuration'])
    ]

    print("\n" + "=" * 150)
    print("CONFRONTO RIASSUNTIVO DEGLI EFFETTI DEI BOUND")
    print("=" * 150)

    print(comparison.round(3).to_string(index=False))

    # Grafico
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        comparison['Configuration'],
        solutions_numeric,
        alpha=0.7
    )

    ax.set_ylabel('# Solutions')
    ax.set_title('Effects of Physical Bounds on AEI Solutions')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, pct in zip(bars, reductions):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{pct:.1f}%',
            ha='center',
            va='bottom'
        )

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

    return comparison