import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd

from .aei_common import mm, HOR, ALPHA_VISC

import sys
sys.path.append("..")
from setup import (
    create_param_grid,
    r_isco, r_horizon, nu_phi, nu_r,
    Rg_SUN, M_BH, NU0, C
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DIAGNOSTICA GENERALIZZATA  (funziona con qualsiasi disk_model)
# ═══════════════════════════════════════════════════════════════════════════

def print_disk_boundaries(disk_model, params, M=M_BH, hr=HOR):
    """
    Stampa un riepilogo delle frontiere di zona per qualsiasi modello di disco.

    Il modello deve restituire un quinto elemento ``info`` con almeno le chiavi:
      ``r_ISCO``, ``r_AB``, ``r_BC``, ``mdot``  (opzionale).

    Se ``info`` non è disponibile, stampa solo i parametri passati.

    Parametri
    ----------
    disk_model : callable
        disk_model(r_rg, **params) → (B0, Sigma, c_s, zone [, info])
    params : dict
        Parametri scalari del disco, es. {'a': 0.5, 'B00': 1e6, 'Sigma0': 1e5}
    M : float
        Massa BH [M_sun]
    hr : float
        Aspect ratio H/r
    """
    a = float(params['a'])
    rISCO = float(r_isco(a))
    rH    = float(r_horizon(a))

    # sonda a r_ISCO per estrarre info
    # firma attuale: (B0, Sigma, c_s, hr, zone, info) → 6 elementi, info=result[5]
    # firma vecchia: (B0, Sigma, c_s, zone, info)     → 5 elementi, info=result[4]
    _r_probe = np.array([rISCO * 1.01])
    result   = disk_model(_r_probe, **params)
    n_ret    = len(result)
    info     = result[5] if n_ret >= 6 else (result[4] if n_ret == 5 else {})

    print(f"\n{'='*64}")
    label_parts = [f"a={a:.2f}"]
    for k, v in params.items():
        if k == 'a':
            continue
        label_parts.append(f"{k}={v:.2g}" if isinstance(v, float) else f"{k}={v}")
    print(f"  DISCO  |  {',  '.join(label_parts)}  |  M={M:.1e} M☉")
    print(f"{'='*64}")
    print(f"  r_H    = {rH:.3f}  r_g")
    print(f"  r_ISCO = {rISCO:.3f}  r_g")

    if 'mdot' in info:
        print(f"  ṁ      = {info['mdot']:.4f}")

    if 'r_AB' in info and 'r_BC' in info:
        r_AB = float(info['r_AB'])
        r_BC = float(info['r_BC'])
        print(f"  r_AB   = {r_AB:.2f}  r_g  ({r_AB/rISCO:.1f} × r_ISCO)")
        print(f"  r_BC   = {r_BC:.1f}  r_g  ({r_BC/r_AB:.1f} × r_AB)")
    else:
        print("  (il modello non espone r_AB / r_BC — aggiornare disk_model"
              " per restituire info come quinto elemento)")

    # eventuali campi extra in info
    _skip = {'r_AB', 'r_BC', 'mdot', 'r_ISCO', 'r_H', 'norms'}
    for k, v in info.items():
        if k in _skip:
            continue
        if isinstance(v, float):
            print(f"  {k:<10} = {v:.4g}")
        elif isinstance(v, (int, str)):
            print(f"  {k:<10} = {v}")

    print(f"{'='*64}")


def scan_disk_grid(
    disk_model,
    param_vectors, extra_params=None,
    M=M_BH, hr=HOR,
):
    """
    Tabella diagnostica su griglia di parametri per qualsiasi modello di disco.
 
    Funziona con qualsiasi ``disk_model`` purché restituisca ``info`` come
    sesto elemento (nuova firma 6-valori) con almeno ``r_AB`` e ``r_BC``.
 
    Parametri
    ----------
    disk_model : callable
        disk_model(r_rg, **params) → (B0, Sigma, c_s, hr, zone, info)
    param_vectors : dict {nome: array_1d}
        Griglia di parametri da esplorare. Il prodotto cartesiano viene
        calcolato internamente.
        Esempi:
          {'a': np.linspace(-0.9,0.9,5), 'Sigma0': np.logspace(0,7,8)}  ← Simple
          {'a': np.linspace(-0.9,0.9,5), 'mdot':   np.logspace(-2,1,8)} ← SS/NT
    extra_params : dict, opzionale
        Parametri fissi aggiuntivi (es. ``{'B00': 1.0}``).
    M : float
        Massa BH [M_sun].
    hr : float
        Aspect ratio H/r.
 
    Restituisce
    -----------
    df : pd.DataFrame
        Colonne: tutte le chiavi di param_vectors, r_ISCO, r_AB, r_BC,
                 r_AB_rISCO, r_BC_rAB, + eventuali campi extra da info.
    """
    if extra_params is None:
        extra_params = {}
 
    keys   = list(param_vectors.keys())
    arrays = [np.asarray(param_vectors[k]) for k in keys]
    grids  = np.meshgrid(*arrays, indexing='ij')
    flat   = {k: g.ravel() for k, g in zip(keys, grids)}
    N_tot  = flat[keys[0]].size
 
    rows = []
    for i in range(N_tot):
        combo  = {k: float(flat[k][i]) for k in keys}
        params = {**combo, **extra_params}
        try:
            rISCO    = float(r_isco(combo['a']))
            _r_probe = np.array([rISCO * 1.01])
            result   = disk_model(_r_probe, **params)
            n_ret    = len(result)
            info     = result[5] if n_ret >= 6 else (result[4] if n_ret == 5 else {})
 
            row = {**combo, 'r_ISCO': rISCO}
            if 'r_AB' in info and 'r_BC' in info:
                r_AB = float(info['r_AB'])
                r_BC = float(info['r_BC'])
                row.update(r_AB=r_AB, r_BC=r_BC,
                           r_AB_rISCO=r_AB / rISCO,
                           r_BC_rAB=r_BC / r_AB)
            else:
                row.update(r_AB=np.nan, r_BC=np.nan,
                           r_AB_rISCO=np.nan, r_BC_rAB=np.nan)
 
            _skip = {'r_AB', 'r_BC', 'r_ISCO', 'r_H', 'norms'}
            for k, v in info.items():
                if k not in _skip and np.isscalar(v):
                    row[k] = v
            rows.append(row)
        except Exception as exc:
            row = {**combo, 'r_ISCO': np.nan,
                   'r_AB': np.nan, 'r_BC': np.nan,
                   'r_AB_rISCO': np.nan, 'r_BC_rAB': np.nan,
                   '_error': str(exc)}
            rows.append(row)
 
    df = pd.DataFrame(rows)
    cols_print = [c for c in df.columns if c != '_error']
    print(df[cols_print].to_string(index=False, float_format='{:.3g}'.format))
    return df
 
 
def plot_boundary_ratios(
    disk_model,
    params,
    r_range=None,
    n_points=500,
    M=M_BH,
    condition_funcs=None,
):
    """
    Diagnostica grafica delle condizioni sulle frontiere di zona.
 
    Plotta le funzioni di crossing (quelle che valgono 1 alla frontiera)
    su un range radiale esteso, mostrando dove avvengono le transizioni.
 
    Parametri
    ----------
    disk_model : callable
        disk_model(r_rg, **params) → (B0, Sigma, c_s, zone [, info])
        Il modello viene chiamato su ogni punto radiale per estrarre B0, Sigma, c_s.
    params : dict
        Parametri scalari del disco.
    r_range : (r_lo, r_hi) o None
        Range radiale [r_g]. Se None: [r_ISCO × 1.01, 1e5].
    n_points : int
        Numero di punti del grid radiale.
    M : float
        Massa BH [M_sun].
    condition_funcs : dict {label: callable(r, B0, Sigma, c_s) -> array} o None
        Funzioni di crossing da plottare.  Se None, si plotta β/(1-β)
        (condizione universale per la frontiera A→B) usando B0 e Sigma
        ricavati da disk_model.
        Ogni callable riceve:
          r     : ndarray [r_g]
          B0    : ndarray [G]
          Sigma : ndarray [g/cm²]
          c_s   : ndarray [cm/s]
        e deve restituire un ndarray con valori adimensionali (crossing a 1).
 
    Restituisce
    -----------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
 
    a = float(params['a'])
    rISCO = float(r_isco(a))
 
    if r_range is None:
        r_lo, r_hi = rISCO * 1.01, 1e5
    else:
        r_lo, r_hi = r_range
 
    r_diag = np.geomspace(r_lo, r_hi, n_points)
 
    # profili fisici
    result    = disk_model(r_diag, **params)
    B0_arr    = result[0]
    Sigma_arr = result[1]
    cs_arr    = result[2]
    n_ret     = len(result)
    # estrai hr(r) dal modello se disponibile (nuova firma 6 valori)
    if n_ret >= 6:
        hr_ret = result[3]
        hr_arr = (np.full_like(r_diag, float(hr_ret))
                  if np.isscalar(hr_ret) else np.asarray(hr_ret, dtype=float))
    else:
        hr_arr = np.full_like(r_diag, float(HOR))
    info      = result[5] if n_ret >= 6 else (result[4] if n_ret == 5 else {})
 
    # condizioni di default: β/(1-β)
    if condition_funcs is None:
        _hr_arr = hr_arr   # cattura nell'closure
        def _beta_ratio(r, B0, Sigma, c_s):
            from .aei_common import compute_beta
            beta = compute_beta(B0, Sigma, c_s, r, _hr_arr, M)
            return beta / np.maximum(1 - beta, 1e-30)
 
        condition_funcs = {r'$\beta\,/\,(1-\beta)$  [A→B]': _beta_ratio}
 
    n_conds = len(condition_funcs)
    fig, axes = plt.subplots(1, n_conds, figsize=(7 * n_conds, 5), squeeze=False)
 
    colors = ['C1', 'C0', 'C2', 'C3', 'C4']
 
    for ax, (label, func), color in zip(axes[0], condition_funcs.items(), colors):
        y = func(r_diag, B0_arr, Sigma_arr, cs_arr)
        y = np.asarray(y, float)
 
        ax.loglog(r_diag, np.abs(y), color=color, lw=2, label=label)
        ax.axhline(1, color='k', ls='--', lw=1, label='= 1  (frontiera)')
        ax.axvline(rISCO, color='gray', ls=':', lw=1, label=f'r_ISCO={rISCO:.1f}')
 
        # linee verticali dalle frontiere note
        for fname, fcolor in [('r_AB', '#f97316'), ('r_BC', '#3b82f6')]:
            if fname in info:
                rv = float(info[fname])
                ax.axvline(rv, color=fcolor, ls='--', lw=1.5,
                           label=f'{fname}={rv:.1f} rg')
 
        # trova crossings numerici
        crossings = r_diag[np.where(np.diff(np.sign(y - 1)))[0]]
        for rc in crossings:
            ax.axvline(rc, color=color, ls=':', lw=1,
                       label=f'crossing≈{rc:.1f} rg')
        if len(crossings) == 0:
            print(f"ATTENZIONE [{label}]: nessun crossing trovato su "
                  f"[{r_lo:.1f}, {r_hi:.0e}] r_g")
            print(f"  valore a r_lo={r_lo:.1f}: {y[0]:.3e}")
            print(f"  valore a r_hi={r_hi:.0e}: {y[-1]:.3e}")
 
        ax.set_xlabel('r [r_g]')
        ax.set_ylabel('valore funzione (crossing @ 1)')
        ax.set_title(f'Condizione: {label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    return fig


def scan_boundary_grid(
    disk_model,
    param_vectors,
    extra_params=None,
    M=M_BH,
):
    """
    Raccoglie r_AB e r_BC su una griglia di parametri e restituisce un DataFrame
    con statistiche e distribuzioni.

    Equivalente generalizzato della cella D1 del notebook (sezione diagnostica v2).

    Parametri
    ----------
    disk_model : callable
        disk_model(r_rg, **params) → (B0, Sigma, c_s, zone [, info])
        Deve restituire ``info`` come quinto elemento con ``r_AB`` e ``r_BC``.
    param_vectors : dict {nome: array_1d}
        Griglia di parametri su cui iterare, es.::

            {
              'a':      np.linspace(-0.9, 0.9, 7),
              'Sigma0': np.logspace(0, 7, 8),
            }

        L'iterazione è il prodotto cartesiano di tutti i vettori.
    extra_params : dict, opzionale
        Parametri fissi aggiuntivi (es. ``{'B00': 1.0, 'alpha_visc': 0.01}``).
    M : float
        Massa BH [M_sun].

    Restituisce
    -----------
    df_bounds : pd.DataFrame
        Colonne: tutte le chiavi di param_vectors, r_ISCO, r_AB, r_BC,
                 r_AB_over_rISCO, r_BC_over_rAB, + eventuali extra da info.
    """
    if extra_params is None:
        extra_params = {}

    keys   = list(param_vectors.keys())
    arrays = [np.asarray(param_vectors[k]) for k in keys]
    grids  = np.meshgrid(*arrays, indexing='ij')
    flat   = {k: g.ravel() for k, g in zip(keys, grids)}
    N      = flat[keys[0]].size

    rows = []
    for i in range(N):
        combo = {k: float(flat[k][i]) for k in keys}
        params = {**combo, **extra_params}
        try:
            rISCO = float(r_isco(combo['a']))
            _r_probe = np.array([rISCO * 1.01])
            result   = disk_model(_r_probe, **params)
            n_ret    = len(result)
            info     = result[5] if n_ret >= 6 else (result[4] if n_ret == 5 else {})

            row = {**combo, 'r_ISCO': rISCO}
            if 'r_AB' in info and 'r_BC' in info:
                r_AB = float(info['r_AB'])
                r_BC = float(info['r_BC'])
                row.update(r_AB=r_AB, r_BC=r_BC,
                           r_AB_over_rISCO=r_AB / rISCO,
                           r_BC_over_rAB=r_BC / r_AB)
            else:
                row.update(r_AB=np.nan, r_BC=np.nan,
                           r_AB_over_rISCO=np.nan, r_BC_over_rAB=np.nan)

            _skip = {'r_AB', 'r_BC', 'r_ISCO', 'r_H', 'norms'}
            for k, v in info.items():
                if k not in _skip and np.isscalar(v):
                    row[k] = v

            rows.append(row)
        except Exception as exc:
            row = {**combo, 'r_ISCO': np.nan,
                   'r_AB': np.nan, 'r_BC': np.nan,
                   'r_AB_over_rISCO': np.nan, 'r_BC_over_rAB': np.nan,
                   '_error': str(exc)}
            rows.append(row)

    df_bounds = pd.DataFrame(rows)

    print("=== Statistiche frontiere ===")
    stat_cols = [c for c in ['r_ISCO','r_AB','r_BC',
                              'r_AB_over_rISCO','r_BC_over_rAB']
                 if c in df_bounds.columns]
    print(df_bounds[stat_cols].describe().round(2))
    return df_bounds


def plot_boundary_distributions(df_bounds, param_color='a'):
    """
    Plotta le distribuzioni di r_AB e r_BC su una griglia di parametri.

    Equivalente generalizzato della cella D2 del notebook (sezione diagnostica v2).

    Parametri
    ----------
    df_bounds : pd.DataFrame
        Output di ``scan_boundary_grid``.
    param_color : str
        Nome della colonna da usare come variabile di colore nei scatter plot.
        Default: 'a' (spin).

    Restituisce
    -----------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    has_rAB = 'r_AB' in df_bounds.columns
    has_rBC = 'r_BC' in df_bounds.columns
    if not (has_rAB and has_rBC):
        raise ValueError("df_bounds deve contenere le colonne r_AB e r_BC.")

    # cerca una colonna "X" per l'asse x degli scatter (la prima non-frontiera)
    _fixed = {'r_ISCO','r_AB','r_BC','r_AB_over_rISCO','r_BC_over_rAB',
              '_error', param_color}
    x_candidates = [c for c in df_bounds.columns
                    if c not in _fixed and df_bounds[c].nunique() > 1]
    x_col = x_candidates[0] if x_candidates else None

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # pannello 1 — istogramma r_AB e r_BC
    axes[0].hist(df_bounds['r_AB'].dropna(), bins=30, alpha=0.7,
                 label='r_AB', color='#f97316')
    axes[0].hist(df_bounds['r_BC'].dropna(), bins=30, alpha=0.7,
                 label='r_BC', color='#3b82f6')
    axes[0].set_xlabel('r [r_g]')
    axes[0].set_ylabel('Conteggio')
    axes[0].set_title('Distribuzione r_AB e r_BC')
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    c_vals = df_bounds[param_color] if param_color in df_bounds.columns else None

    # pannello 2 — r_AB vs x_col
    if x_col:
        sc = axes[1].scatter(df_bounds[x_col], df_bounds['r_AB'],
                             c=c_vals, cmap='RdBu', s=15, alpha=0.7)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel(x_col)
        axes[1].set_ylabel('r_AB [r_g]')
        axes[1].set_title(f'r_AB vs {x_col}  (colore = {param_color})')
        plt.colorbar(sc, ax=axes[1], label=param_color)
        axes[1].grid(True, alpha=0.2)

        sc2 = axes[2].scatter(df_bounds[x_col], df_bounds['r_BC'],
                              c=c_vals, cmap='RdBu', s=15, alpha=0.7)
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].set_xlabel(x_col)
        axes[2].set_ylabel('r_BC [r_g]')
        axes[2].set_title(f'r_BC vs {x_col}  (colore = {param_color})')
        plt.colorbar(sc2, ax=axes[2], label=param_color)
        axes[2].grid(True, alpha=0.2)
    else:
        # scatter r_AB vs r_BC
        sc = axes[1].scatter(df_bounds['r_AB'], df_bounds['r_BC'],
                             c=c_vals, cmap='RdBu', s=15, alpha=0.7)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('r_AB [r_g]')
        axes[1].set_ylabel('r_BC [r_g]')
        axes[1].set_title(f'r_AB vs r_BC  (colore = {param_color})')
        plt.colorbar(sc, ax=axes[1], label=param_color)
        axes[1].grid(True, alpha=0.2)

        # r_AB/r_ISCO histogram
        axes[2].hist(df_bounds['r_AB_over_rISCO'].dropna(), bins=30, alpha=0.7, color='#f97316')
        axes[2].set_xlabel('r_AB / r_ISCO')
        axes[2].set_ylabel('Conteggio')
        axes[2].set_title('Separazione frontiera A-B da ISCO')
        axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    return fig