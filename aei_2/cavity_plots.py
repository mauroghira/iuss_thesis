"""
cavity_plots.py
===============
Funzioni di plotting aggiuntive per l'analisi AEI cavity (Simple-v1).

Da importare in radial_diagnosis.py o usare standalone dopo aver eseguito
run_all() per ottenere all_results.

Funzioni
--------
  plot_paramspace_heatmap   Versione colormap di cavity_paramspace.png:
                            3 pannelli (shear / beta / WKB) con heatmap
                            della frazione di spin con soluzione, più
                            linee di riferimento e croce sui valori ref.

  plot_dQdr_at_CR           dQ/dr valutata a r_CR in funzione di spin,
                            per m=1 e m=2 sullo stesso plot (due curve).

  plot_k_violin_by_spin     Violin plot della distribuzione di k per bin
                            di spin (suggerimento 4 del file originale),
                            un pannello per ogni configurazione (m, H/r).

Dipendenze
----------
  Stesse del file radial_diagnosis.py originale.
  Assumiamo che le variabili globali CONFIGS, A_GRID, B00_GRID,
  SIGMA0_GRID, B00_REF, SIGMA0_REF, N_R siano importate o ridefinite
  nello script che chiama queste funzioni.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec


# ── costanti condivise (ricopiate per portabilità) ──────────────────────────
from aei_setup import (
    compute_dQdr, r_ilr, r_olr, r_corotation, 
    _make_interp, make_disk, get_resonances, solve_k_aei,
    ALP_B, ALP_S, CONFIGS , A_GRID, B00_GRID, 
    SIGMA0_GRID, B00_REF, SIGMA0_REF, N_R, N_R_CAV
)

import sys
sys.path.append('..')

from setup import M_BH, NU0, r_isco, set_style, fix_spines
from aei_2.simple_disc import disk_model_simple

set_style()

# ═══════════════════════════════════════════════════════════════════════════════
# 1 ── PARAMSPACE HEATMAP  (versione colormap di cavity_paramspace.png)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_paramspace_heatmap(all_results, cfg_index=0, figsize=(7, 2.2)):
    """
    Versione colormap di cavity_paramspace.png.

    Tre pannelli (shear / +beta / +WKB), ciascuno mostra la heatmap 2D
    nel piano (Sigma0, B00) della *frazione di spin* per cui quella
    combinazione di parametri passa il filtro.

    Aggiunte rispetto alla versione scatter originale:
      • colormap RdYlGn (come cavity_summary)
      • linea verticale a Sigma0 = SIGMA0_REF
      • linea orizzontale a B00 = B00_REF
      • croce (×) sul punto (SIGMA0_REF, B00_REF)

    Parameters
    ----------
    all_results : dict  {(mm, hor): pd.DataFrame}
        Output di run_all().
    cfg_index   : int
        Indice in CONFIGS della configurazione da plottare (default 0 = m=1, H/r=0.05).
    figsize     : tuple
    """
    cfg = CONFIGS[cfg_index]
    mm, hor = cfg['mm'], cfg['hor']
    df = all_results[(mm, hor)]

    stages = [
        ('pass_shear', r'$dQ/dr > 0$ at $r_{\rm CR}$'),
        ('pass_beta',  r'$+\ \beta \leq 1$ everywhere'),
        ('pass_wkb',   r'$+$ WKB in cavity'),
    ]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, figure=fig,
                       width_ratios=[1, 1, 1, 0.08],
                       wspace=0.05)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax  = fig.add_subplot(gs[0, 3])   # asse dedicato alla colorbar
    
    for ax, (col, title) in zip(axes, stages):
        fix_spines(ax)
        # ── build heatmap matrix ───────────────────────────────────────────
        frac_map = np.full((len(B00_GRID), len(SIGMA0_GRID)), np.nan)
        for i, B00 in enumerate(B00_GRID):
            for j, S0 in enumerate(SIGMA0_GRID):
                sub = df[(df['B00'] == B00) & (df['Sigma0'] == S0)]
                if len(sub) == 0:
                    continue
                if col not in df.columns:
                    frac_map[i, j] = 0.0
                    continue
                frac_map[i, j] = sub[col].sum() / len(sub)

        # ── plot ───────────────────────────────────────────────────────────
        im = ax.pcolormesh(SIGMA0_GRID, B00_GRID, frac_map,
                           vmin=0, vmax=1, cmap='RdYlGn', shading='auto')

        # ── reference lines and cross ──────────────────────────────────────
        ax.axvline(SIGMA0_REF, color='white', ls='--', lw=1, alpha=0.85)
        ax.axhline(B00_REF,   color='white', ls='--', lw=1, alpha=0.85)
        ax.plot(SIGMA0_REF, B00_REF, 'x',
                color='white', ms=2, mew=1.5, zorder=6)

        # annotation: total fraction for this filter
        if col in df.columns:
            n_pass = df[col].sum()
            n_tot  = len(df)
            ax.text(0.03, 0.97,
                    f'{n_pass}/{n_tot} ({100*n_pass/n_tot:.1f}%)',
                    transform=ax.transAxes, va='top',
                    color='white',
                    bbox=dict(boxstyle='round', fc='black', alpha=0.5))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\Sigma_0$  [g/cm²]')
        ax.set_title(title)

        # Nascondi le etichette y dei pannelli interni
        if col != 'pass_shear':
            ax.tick_params(labelleft=False)
            ax.sharey(axes[0])

    axes[0].set_ylabel(r'$B_{00}$  [G]')

    # Colorbar sull'asse dedicato → non tocca la larghezza dei pannelli
    norm = Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='RdYlGn'), cax=cax)
    cbar.set_label("Frequency [Hz]")

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'cavity_paramspace_heatmap_{hor}_{mm}.{ext}',
                    bbox_inches='tight', dpi=150)
    print('  -> cavity_paramspace_heatmap.pdf / .png')
    plt.show()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2 ── dQ/dr AT COROTATION VS SPIN  (suggerimento 1)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dQdr_at_CR(hor=0.05, a_grid=None, figsize=(3.4, 3.1)):
    """
    dQ/dr valutata a r_CR in funzione dello spin, per m=1 e m=2.

    Il segno di dQ/dr alla corotazione determina se la condizione di shear
    è soddisfatta: dQ/dr(r_CR) > 0 ↔ shear OK.
    Indipendente da B00, Sigma0 (fattore moltiplicativo costante).

    Parameters
    ----------
    hor    : float  aspect ratio (non entra in Q, ma serve per make_disk)
    a_grid : array-like  spin grid (default A_GRID)
    figsize: tuple
    """
    if a_grid is None:
        a_grid = A_GRID

    # configurazioni m=1 e m=2 (un solo H/r — il segno non dipende da hor)
    m_configs = [
        dict(mm=1, color='#3b82f6', label=r'$m=1$'),
        dict(mm=2, color='#f97316', label=r'$m=2$'),
    ]

    fig, ax = plt.subplots(figsize=figsize)
    fix_spines(ax)

    for mcfg in m_configs:
        mm = mcfg['mm']
        if mm==2:
            continue
        dq_cr_vals = []
        a_valid    = []

        for a_val in a_grid:
            rISCO = float(r_isco(a_val))
            _, _, r_CR = get_resonances(a_val, mm)

            if not np.isfinite(r_CR) or r_CR <= rISCO * 1.001:
                dq_cr_vals.append(np.nan)
                a_valid.append(a_val)
                continue

            # evaluate dQ/dr in a narrow interval around r_CR
            r_eps = np.linspace(r_CR * 0.92, r_CR * 1.08, 40)
            res   = make_disk(r_eps, a_val, B00_REF, SIGMA0_REF, hor)
            B0i   = _make_interp(r_eps, res[0])
            Si    = _make_interp(r_eps, res[1])
            dq    = compute_dQdr(r_eps, a_val, B0i, Si, M_BH)

            dq_CR = float(np.interp(r_CR, r_eps, dq))
            dq_cr_vals.append(dq_CR)
            a_valid.append(a_val)

        dq_arr = np.array(dq_cr_vals)
        a_arr  = np.array(a_valid)

        # normalise by a robust amplitude so both curves are comparable
        amp = np.nanpercentile(np.abs(dq_arr), 90)
        amp = amp if amp > 0 else 1.0

        ax.plot(a_arr, dq_arr / amp,
                color=mcfg['color'], lw=1, label=mcfg['label'])

        # shade the region dQ/dr > 0  (shear OK)
        ax.fill_between(a_arr, 0, dq_arr / amp,
                        where=(dq_arr > 0),
                        color=mcfg['color'], alpha=0.12)

        # mark zero crossings with vertical dashed lines
        crossings = np.where(np.diff(np.sign(dq_arr)))[0]
        for ci in crossings:
            ax.axvline(a_arr[ci], color=mcfg['color'],
                       ls=':', lw=1, alpha=0.7)

    ax.set_xlabel('$a$')
    ax.set_ylabel(r'$dQ/dr\,|_{r_{\rm CR}}$ ',)
    #ax.set_title(r'Shear condition at corotation: $dQ/dr\,(r_{\rm CR})$ vs spin',)
    ax.legend()

    # annotation: positive = shear OK
    ax.text(0.98, 0.97, r'$dQ/dr > 0$: shear OK',
            transform=ax.transAxes, ha='right', va='top', color='white',
            bbox=dict(boxstyle='round', fc='black', alpha=0.4))

    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'dQdr_CR.{ext}', bbox_inches='tight', dpi=150)
    print('  -> dQdr_at_CR.pdf / .png')
    plt.show()
    return fig
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# C ── WKB HEATMAP COMPARISON  (B00 × Sigma0, due H/r affiancati)
# ═══════════════════════════════════════════════════════════════════════════════
 
def plot_wkb_heatmap_comparison(all_results, mm=1, figsize=(7, 2.5)):
    """
    Opzione C — Due pannelli affiancati della stessa heatmap (B00, Sigma0)
    colorata per *frazione di spin WKB-valid*, per H/r=0.05 e H/r=0.001.
 
    Il pannello H/r=0.05 appare completamente rosso (nessuna soluzione);
    il pannello H/r=0.001 mostra le regioni verdi dove le soluzioni esistono.
    Il contrasto immediato rende evidente il ruolo di H/r.
 
    Aggiunge linee di riferimento e croce su (SIGMA0_REF, B00_REF).
 
    Parameters
    ----------
    all_results : dict  output di run_all()
    mm          : int   modo azimutale (filtra i 2 pannelli con questo m)
    figsize     : tuple
    """
    cfgs_m = [c for c in CONFIGS if c['mm'] == mm]
    if len(cfgs_m) < 2:
        print(f'Serve almeno 2 configurazioni con mm={mm}. Trovate: {len(cfgs_m)}')
        return None
 
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 3, figure=fig,
                       width_ratios=[1, 1, 0.08],
                       wspace=0.05)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cax  = fig.add_subplot(gs[0, 2])   # asse dedicato alla colorbar
 
    # shared colour scale 0→1
    vmin, vmax = 0.0, 1.0
 
    for ax, cfg in zip(axes, cfgs_m):
        fix_spines(ax)
        hor = cfg['hor']
        key = (mm, hor)
 
        if key not in all_results:
            ax.text(0.5, 0.5, f'Dati mancanti\n(mm={mm}, hor={hor})',
                    ha='center', va='center', transform=ax.transAxes)
            continue
 
        df = all_results[key]
 
        # ── build heatmap ──────────────────────────────────────────────────
        frac_map = np.full((len(B00_GRID), len(SIGMA0_GRID)), np.nan)
        for i, B00 in enumerate(B00_GRID):
            for j, S0 in enumerate(SIGMA0_GRID):
                sub = df[(df['B00'] == B00) & (df['Sigma0'] == S0)]
                if len(sub) == 0:
                    continue
                frac_map[i, j] = sub['pass_wkb'].sum() / len(sub)
 
        n_valid_cells = int(np.sum(frac_map > 0))
        n_cells       = frac_map.size
 
        im = ax.pcolormesh(
            SIGMA0_GRID, B00_GRID, frac_map,
            vmin=vmin, vmax=vmax,
            cmap='RdYlGn', shading='auto',
        )
 
        # ── reference lines and cross ──────────────────────────────────────
        ax.axvline(SIGMA0_REF, color='white', ls='--', lw=1.5, alpha=0.9)
        ax.axhline(B00_REF,   color='white', ls='--', lw=1.5, alpha=0.9)
        ax.plot(SIGMA0_REF, B00_REF, 'x',
                color='white', ms=12, mew=2.5, zorder=6)
        # labels for the reference lines
        ax.text(SIGMA0_REF * 1.05, B00_GRID[1],
                rf'$\Sigma_0={SIGMA0_REF:.0e}$',
                color='white', va='bottom')
        ax.text(SIGMA0_GRID[1], B00_REF * 1.15,
                rf'$B_{{00}}={B00_REF:.0e}$',
                color='white', va='bottom')
 
        # ── stats annotation ───────────────────────────────────────────────
        frac_overall = df['pass_wkb'].sum() / len(df) if len(df) > 0 else 0.0
        ax.text(0.03, 0.97,
                f'Overall: {frac_overall*100:.1f}% valid\n'
                f'{n_valid_cells}/{n_cells} cells > 0',
                transform=ax.transAxes, va='top',
                color='white',
                bbox=dict(boxstyle='round', fc='black', alpha=0.5))
 
        k_max = 1.0 / hor
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\Sigma_0$  [g/cm²]')
        ax.set_ylabel(r'$B_{00}$  [G]')
        ax.set_title(
            rf'$H/r = {hor}$  ($k_{{\rm max}} = {k_max:.0f}$)'
            f'\nWKB-valid fraction  —  $m={mm}$',
        )
 
    fig.suptitle(
        rf'WKB validity: $H/r=0.05$ vs $H/r=10^{{-3}}$  ($m={mm}$)'
        '\nFraction of spins passing WKB in cavity $[r_{{\\rm ISCO}},\\ r_{{\\rm ILR}}]$',
    )
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        plt.savefig(f'wkb_heatmap_comparison.{ext}', bbox_inches='tight', dpi=150)
    print('  -> wkb_heatmap_comparison.pdf / .png')
    plt.show()
    return fig