"""
aei_common.py
=============
Funzioni comuni a tutti i modelli AEI (simple_disc, full_disk_SS, NT).

Contiene:
  - solve_k_aei        : solver della relazione di dispersione (Tagger & Pellat 1999, Eq.17)
  - check_k_wkb        : vincolo WKB su k adimensionale
  - check_beta_aei     : β ≤ 1 (pressione magnetica dominante)
  - check_shear_aei    : dQ/dr > 0 (condizione di shear)
  - compute_beta       : calcolo di β
  - compute_dQdr       : calcolo di dQ/dr con differenze finite
  - find_rossby        : finder vettorizzato, funziona con qualsiasi modello di disco

──────────────────────────────────────────────────────────────────────────────
NOTE SULLE UNITÀ (da Tagger & Pellat 1999)
──────────────────────────────────────────────────────────────────────────────
Il paper lavora nella variabile  s = ln(r),  quindi il numero d'onda k
è definito come:
    ∂/∂s Φ_M ≈ ik Φ_M   →   k adimensionale

Il numero d'onda fisico radiale è  k_fis = k / r_cm  [1/cm].

Analisi dimensionale dell'Eq. (17):

  Termine magnetico:  2B₀²/Σ · (k/r)
    [G²/(g/cm²)] · [1/cm]  =  [cm·s⁻²] · [1/cm]  =  s⁻²  ✓   (r in cm)

  Termine pressione:  k²/r² · c_s²
    [adim²/cm²] · [cm²/s²]  =  s⁻²  ✓                          (r in cm)

Quindi i coefficienti A, B della quadratica devono usare r in cm.
Il k restituito dal solver è adimensionale.

Conseguenze sui check fisici:
  Il WKB richiede  k_fis ≫ 1/r, ovvero  k_fis · r ≫ 1.
  Poiché  k_fis · r_cm = k  (adimensionale), il bound WKB si traduce in:
      k_min < k < k_max    (tutto adimensionale)
  con k_min = 0.1, k_max = 10  (come da notebook, sezione AEI setup).

──────────────────────────────────────────────────────────────────────────────
STRUTTURA DEL DISCO — interfaccia per find_rossby
──────────────────────────────────────────────────────────────────────────────
find_rossby accetta un callable  `disk_model`  con firma:

    B0, Sigma, c_s = disk_model(r_rg, **extra_params)

dove r_rg è un array 1D di raggi in r_g e **extra_params è qualsiasi
set di parametri scalari (a, B00, Sigma0, alpha_visc, …).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys
sys.path.append("..")
from setup import (
    r_isco, nu_phi, nu_r, r_horizon,
    Rg_SUN, M_BH, NU0,
)

HOR = 0.05
mm = 1
ALPHA_VISC = 0.01

# ═══════════════════════════════════════════════════════════════════════════
# 1.  SOLVER  (identico per tutti i modelli)
# ═══════════════════════════════════════════════════════════════════════════

def solve_k_aei(r_rg, a, B0, Sigma, c_s, m=mm, M=M_BH):
    """
    Risolve la relazione di dispersione AEI (Tagger & Pellat 1999, Eq. 17):

        ω̃² = κ² + (2B₀²/Σ)(k/r) + (k²/r²) c_s²

    con  ω̃ = ω_obs − m Ω_φ,  k adimensionale (numero d'onda in s = ln r).

    La quadratica in k è:
        A k² + B k + C = 0
        A = c_s² / r_cm²
        B = 2 B₀² / (Σ r_cm)
        C = κ² − ω̃²

    Parametri
    ----------
    r_rg  : array_like   raggio in unità di r_g  (adimensionale)
    a     : float        spin del BH  [adim, −1…1]
    B0    : array_like   campo magnetico verticale  [G]
    Sigma : array_like   densità superficiale  [g/cm²]
    c_s   : array_like   velocità del suono  [cm/s]
    m     : int          numero d'onda azimutale  (default 1)
    M     : float        massa BH  [M_sun]

    Restituisce
    -----------
    k : ndarray
        Numero d'onda adimensionale  (NaN dove non esiste soluzione reale > 0).
        Relazioni utili:
          k_fisico [1/cm]  =  k / (r_rg * Rg)
          k_adimensionale    [adim]  =  k                  (≡ k_fis × r_cm)
    """
    r_rg  = np.asarray(r_rg,  dtype=float)
    B0    = np.asarray(B0,    dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    c_s   = np.asarray(c_s,   dtype=float)

    Rg   = Rg_SUN * M
    r_cm = r_rg * Rg                              # cm — richiesto dalla dim. analysis

    kappa_sq  = (2 * np.pi * nu_r(r_rg,  a, M))**2
    Omega_phi = 2 * np.pi * nu_phi(r_rg, a, M)
    om_tilde  = 2 * np.pi * NU0 - m * Omega_phi  # ω_obs − m Ω_φ

    # Coefficienti  A k² + B k + C = 0  (tutti in s⁻²)
    A  = c_s**2   / r_cm**2
    B  = 2*B0**2  / (Sigma * r_cm)
    C  = kappa_sq - om_tilde**2

    Delta = B**2 - 4*A*C

    k = np.full_like(r_rg, np.nan)
    ok = Delta >= 0
    if np.any(ok):
        sqD      = np.sqrt(Delta[ok])
        kp       = (-B[ok] + sqD) / (2*A[ok])
        km       = (-B[ok] - sqD) / (2*A[ok])
        # preferisci il ramo k_plus (più grande); fallback su k_minus
        k[ok] = np.where(kp > 0, kp, np.where(km > 0, km, np.nan))

    return k   # adimensionale


# ═══════════════════════════════════════════════════════════════════════════
# 2.  CHECK FISICI  (tutti operano su k adimensionale)
# ═══════════════════════════════════════════════════════════════════════════

def check_k_wkb(k, k_min=0.1, k_max=10.0):
    """
    Vincolo WKB sul numero d'onda adimensionale k = k_fis · r_cm.

    Il limite WKB richiede  k_fis ≫ 1/r,  cioè  k ≫ 1.
    In pratica si usa il range conservativo  [0.1, 10]

    Parametri
    ----------
    k     : array_like   numero d'onda adimensionale
    k_min : float        limite inferiore (default 0.1)
    k_max : float        limite superiore (default 10.0)

    Restituisce
    -----------
    mask : ndarray bool
    """
    k = np.asarray(k)
    return np.isfinite(k) & (k >= k_min) & (k <= k_max)


def compute_beta(B0, Sigma, c_s, r_rg, hr=HOR, M=M_BH):
    """
    Parametro plasma beta:

        β = 8π Σ c_s² / (H B₀²)    con H = hr · r · Rg

    Parametri
    ----------
    B0, Sigma, c_s : array_like   profili fisici  [G, g/cm², cm/s]
    r_rg  : array_like            raggio  [r_g]
    hr    : float                 aspect ratio H/r
    M     : float                 massa BH  [M_sun]

    Restituisce
    -----------
    beta : ndarray
    """
    Rg = Rg_SUN * M
    H  = hr * np.asarray(r_rg) * Rg
    return 8*np.pi * np.asarray(Sigma) * np.asarray(c_s)**2 / (H * np.asarray(B0)**2)


def check_beta_aei(B0, Sigma, c_s, r_rg, hr=HOR, M=M_BH, beta_max=1.0):
    """
    Condizione AEI: β ≤ beta_max  (pressione magnetica dominante).

    Restituisce
    -----------
    mask : ndarray bool
    """
    return compute_beta(B0, Sigma, c_s, r_rg, hr, M) <= beta_max


def compute_dQdr(r_rg, a, B0_func, Sigma_func, M=M_BH, dr_frac=0.01):
    """
    Derivata radiale di  Q = Ω_φ · Σ / B₀²,  con differenze finite.

    Condizione AEI: dQ/dr > 0.

    Parametri
    ----------
    r_rg      : array_like   raggi  [r_g]
    a         : float        spin
    B0_func   : callable     B0_func(r_rg)  → B0  [G]
    Sigma_func: callable     Sigma_func(r_rg) → Sigma  [g/cm²]
    M         : float        massa  [M_sun]
    dr_frac   : float        passo relativo per la diff. finita

    Restituisce
    -----------
    dQdr : ndarray   [unità di 1/r_g, relative a Ω·Σ/B²]
    """
    r_rg = np.asarray(r_rg, dtype=float)
    dr   = dr_frac * r_rg
    rp   = r_rg + dr

    def Q(r):
        return 2*np.pi * nu_phi(r, a, M) * Sigma_func(r) / B0_func(r)**2

    return (Q(rp) - Q(r_rg)) / dr


def check_shear_aei(r_rg, a, B0_func, Sigma_func, M=M_BH, dr_frac=0.01):
    """
    Condizione di shear AEI: dQ/dr > 0.

    Parametri: stessi di compute_dQdr.

    Restituisce
    -----------
    mask : ndarray bool
    """
    return compute_dQdr(r_rg, a, B0_func, Sigma_func, M, dr_frac) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 3.  FINDER VETTORIZZATO
# ═══════════════════════════════════════════════════════════════════════════

def find_rossby(
    r_vec,
    param_grid,
    disk_model,
    m=mm,
    hr=HOR,
    M=M_BH,
    check_k=True,
    check_beta=True,
    check_shear=True,
    k_min=0.1,
    k_max=10.0,
    beta_max=1.0,
    dr_frac=0.01,
):
    """
    Finder vettorizzato della relazione di dispersione AEI.

    Funziona con qualsiasi modello di disco purché venga fornito
    l'adapter `disk_model` (vedi sotto).

    ──────────────────────────────────────────────────────────────────
    Strategia di vettorizzazione
    ──────────────────────────────────────────────────────────────────
    L'ostacolo alla piena vettorizzazione è che per i modelli SS e NT
    le frontiere  r_AB, r_BC  dipendono da (a, Sigma0) e non da r:
    non si possono precalcolare in un unico meshgrid.

    Soluzione adottata:
      - si itera sul prodotto cartesiano dei parametri *fissi per disco*
        (tutto tranne r),  che è in genere N_a × N_B00 × N_Sigma0 ≪ N_tot
      - per ogni combinazione, si chiama disk_model(r_vec, **params)
        che restituisce (B0, Sigma, c_s) sull'intero vettore r — questa
        chiamata è completamente vettorizzata in r
      - i check fisici e il solver sono anch'essi vettorizzati in r

    Costo: O(N_a × N_B00 × N_Sigma0 × N_r)  con solo  N_a × N_B00 × N_Sigma0
    chiamate Python (contro  N_a × N_B00 × N_Sigma0 × N_r  nel vecchio codice).

    ──────────────────────────────────────────────────────────────────
    Parametri
    ──────────────────────────────────────────────────────────────────
    r_vec      : array_like
        Raggi in r_g su cui calcolare i profili.

    param_grid : dict  {nome: array_1d}
        Griglia dei parametri *che non dipendono da r*.
        Deve contenere almeno 'a'.
        Esempio per simple_disc:   {'a': ..., 'B00': ..., 'Sigma0': ...}
        Esempio per full_disk_SS:  {'a': ..., 'B00': ..., 'Sigma0': ...}
        (i parametri 'r' vanno in r_vec, non qui)

    disk_model : callable
        disk_model(r_rg, **row) → (B0, Sigma, c_s)
        dove row è un dict con i parametri scalari della riga corrente.
        Vedi sezione adapter in fondo al file.

    m, hr, M   : int, float, float
        Modo azimutale, aspect ratio, massa BH.

    check_k, check_beta, check_shear : bool
        Attiva/disattiva i check fisici.

    k_min, k_max : float
        Range WKB per k adimensionale.

    beta_max : float
        Soglia per il check β.

    dr_frac : float
        Passo relativo per dQ/dr.

    ──────────────────────────────────────────────────────────────────
    Restituisce
    ──────────────────────────────────────────────────────────────────
    df : DataFrame
        Una riga per ogni (r, **params) che supera tutti i check attivi.
        Colonne garantite:
          r, a, k, beta, dQdr
          + tutte le chiavi di param_grid
          + 'zone' (se disk_model la restituisce, altrimenti assente)
    """
    r_vec = np.asarray(r_vec, dtype=float)

    # ── prodotto cartesiano dei parametri non-r ──────────────────────────────
    keys   = list(param_grid.keys())
    arrays = [np.asarray(param_grid[k]) for k in keys]

    # meshgrid N-D sui parametri non-r
    grids = np.meshgrid(*arrays, indexing='ij')
    flat  = {k: g.ravel() for k, g in zip(keys, grids)}
    N_combos = flat[keys[0]].size

    # ── vincolo ISCO (dipende solo da a) ────────────────────────────────────
    a_flat    = flat['a']
    isco_flat = r_isco(a_flat)          # (N_combos,)

    rows = []

    for i in range(N_combos):
        # parametri scalari per questa combinazione
        row_params = {k: float(flat[k][i]) for k in keys}
        a_val      = row_params['a']
        isco_val   = float(isco_flat[i])

        # maschera ISCO su r
        r_ok = r_vec >= isco_val
        r_i  = r_vec[r_ok]
        if r_i.size == 0:
            continue

        # ── profili fisici (vettorizzati in r) ──────────────────────────────
        result = disk_model(r_i, **row_params)
        n_ret  = len(result)
        B0_i, Sigma_i, cs_i = result[0], result[1], result[2]
        zone_i = result[3] if n_ret >= 4 else None
        info_i = result[4] if n_ret == 5 else {} # in realtà non serge

        # ── solver k ────────────────────────────────────────────────────────
        k_i = solve_k_aei(r_i, a_val, B0_i, Sigma_i, cs_i, m=m, M=M)

        # ── check fisici ────────────────────────────────────────────────────
        mask = np.isfinite(k_i) & (k_i > 0)

        if check_k:
            mask &= check_k_wkb(k_i, k_min, k_max)

        # beta e dQdr calcolati solo dove k è già valido (risparmio CPU)
        beta_i = np.full_like(r_i, np.nan)
        dQdr_i = np.full_like(r_i, np.nan)

        if np.any(mask):
            beta_i[mask] = compute_beta(
                B0_i[mask], Sigma_i[mask], cs_i[mask], r_i[mask], hr, M
            )

            # per dQdr serve B0_func e Sigma_func come callable in r
            # le costruiamo interpolando i profili già calcolati sull'intero r_i
            # (evita di richiamare disk_model due volte)
            _B0_interp    = _make_interp(r_i, B0_i)
            _Sigma_interp = _make_interp(r_i, Sigma_i)
            dQdr_i[mask]  = compute_dQdr(
                r_i[mask], a_val, _B0_interp, _Sigma_interp, M, dr_frac
            )

        if check_beta:
            mask &= (beta_i <= beta_max)
        if check_shear:
            mask &= (dQdr_i > 0)

        # ── raccolta risultati ───────────────────────────────────────────────
        if not np.any(mask):
            continue

        idx_ok = np.where(mask)[0]
        for j in idx_ok:
            entry = dict(row_params)
            entry['r']     = r_i[j]
            entry['k']     = k_i[j]
            entry['beta']  = beta_i[j]
            entry['dQdr']  = dQdr_i[j]
            entry['m']     = m
            entry['hr']    = hr
            if zone_i is not None:
                entry['zone'] = zone_i[j]
            rows.append(entry)

    return pd.DataFrame(rows)


# ── helper interno: interpolazione lineare log-log per ricostruire profili ──

def _make_interp(r_nodes, y_nodes):
    """
    Restituisce una funzione  f(r) → y  tramite interpolazione lineare
    in spazio log-log (power-law locale tra i nodi).
    Usato internamente da find_rossby per calcolare dQ/dr senza
    richiamare disk_model una seconda volta.
    """
    log_r = np.log(r_nodes)
    log_y = np.log(np.maximum(y_nodes, 1e-300))

    def interp(r_query):
        r_query = np.asarray(r_query, dtype=float)
        return np.exp(np.interp(np.log(r_query), log_r, log_y))

    return interp


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PLOT e stats
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


# ═══════════════════════════════════════════════════════════════════════════
# 5.  FULL DISC INFOS PER ANALISI RADIALE
# ═══════════════════════════════════════════════════════════════════════════
def compute_disk_profile(
    disk_model,
    params,
    mm=mm,
    hr=HOR,
    M=M_BH,
    r_min=None,
    r_max=None,
    n_points=300,
):
    """
    Profilo radiale completo per qualsiasi modello di disco.

    Funziona con la stessa `disk_model` passata a `find_rossby` —

    Parametri
    ----------
    disk_model : callable
        disk_model(r_rg, **params) → (B0, Sigma, c_s, zone)
        Stessa firma usata in find_rossby.
        Può restituire un quinto elemento `info` (dict con r_ISCO, r_AB,
        r_BC, mdot, ...) che viene incluso in meta automaticamente.

    params : dict
        Parametri scalari del disco, es.:
          {'a': 0.5, 'B00': 1e6, 'Sigma0': 1e5}
        Vengono passati a disk_model come **kwargs.

    mm : int
        Modo azimutale m della perturbazione AEI.

    hr : float
        Aspect ratio H/r (per i check β e c_s se non già in disk_model).

    M : float
        Massa BH [M_sun].

    r_min : float o None
        Raggio interno della griglia [r_g].
        Se None: r_isco(params['a'])

    r_max : float o None
        Raggio esterno della griglia [r_g].
        Se None: richiede che disk_model restituisca info['r_BC'],
                 allora r_max = 3 × r_BC.
                 Se info non è disponibile, solleva ValueError.

    n_points : int
        Numero di punti radiali log-spaziati.

    Restituisce
    -----------
    df : pd.DataFrame
        Colonne: r, zone, B0, Sigma, c_s, k, beta, dQdr,
                 k_valid, beta_valid, shear_valid, aei_valid

    meta : dict
        Contiene almeno: r_H, r_ISCO, a, mm, hr, M, + tutto ciò che
        disk_model restituisce in info (r_AB, r_BC, mdot, norms, ...).

    Esempi
    ------
    # modello SS
    df, meta = compute_disk_profile(
        disk_model = lambda r, **p: disk_model_SS(r, **p, alpha_visc=0.1, hr=0.05),
        params     = {'a': 0.5, 'B00': 1e6, 'Sigma0': 1e5},
        mm=1, hr=0.05,
    )

    # modello NT — identico, solo disk_model cambia
    df, meta = compute_disk_profile(
        disk_model = lambda r, **p: disk_model_NT(r, **p, alpha_visc=0.1, hr=0.05),
        params     = {'a': 0.5, 'B00': 1e6, 'Sigma0': 1e5},
        mm=1, hr=0.05,
    )

    # con r_min/r_max espliciti
    df, meta = compute_disk_profile(
        disk_model = my_model,
        params     = {'a': 0.5, 'alpha': 0.3},
        mm=1, hr=0.05,
        r_min=5.0, r_max=500.0,
    )
    """
    a = float(params['a'])

    # ── griglia radiale ───────────────────────────────────────────────────
    rISCO = float(r_isco(a))
    rH    = float(r_horizon(a))

    if r_min is None:
        r_min = rISCO

    # Chiamata di prova su un singolo punto per estrarre info se disponibile
    _r_probe = np.array([r_min])
    _result  = disk_model(_r_probe, **params)
    _info    = _result[4] if len(_result) == 5 else {}

    if r_max is None:
        if 'r_BC' in _info:
            r_max = 3.0 * _info['r_BC']
        else:
            raise ValueError(
                "r_max non specificato e disk_model non restituisce info['r_BC']. "
                "Passare r_max esplicitamente oppure aggiornare disk_model per "
                "restituire un quinto elemento info={'r_BC': ...}."
            )

    r_arr = np.geomspace(r_min, r_max, n_points)

    # ── profili fisici ────────────────────────────────────────────────────
    result    = disk_model(r_arr, **params)
    n_ret     = len(result)
    B0_arr    = np.asarray(result[0], dtype=float)
    Sigma_arr = np.asarray(result[1], dtype=float)
    cs_arr    = np.asarray(result[2], dtype=float)
    zone_arr  = np.asarray(result[3]) if n_ret >= 4 else np.full(len(r_arr), 'N/A', dtype=object)
    info      = result[4] if n_ret == 5 else {}

    # ── solver AEI ────────────────────────────────────────────────────────
    k_arr    = solve_k_aei(r_arr, a, B0_arr, Sigma_arr, cs_arr, m=mm, M=M)
    beta_arr = compute_beta(B0_arr, Sigma_arr, cs_arr, r_arr, hr, M)
    dQdr_arr = compute_dQdr(r_arr, a,
                            _make_interp(r_arr, B0_arr),
                            _make_interp(r_arr, Sigma_arr), M)

    # ── maschere di validità ──────────────────────────────────────────────
    k_valid     = check_k_wkb(k_arr)
    beta_valid  = beta_arr <= 1.0
    shear_valid = dQdr_arr > 0
    aei_valid   = k_valid & beta_valid & shear_valid

    df = pd.DataFrame({
        'r':           r_arr,
        'zone':        zone_arr,
        'B0':          B0_arr,
        'Sigma':       Sigma_arr,
        'c_s':         cs_arr,
        'k':           k_arr,
        'beta':        beta_arr,
        'dQdr':        dQdr_arr,
        'k_valid':     k_valid,
        'beta_valid':  beta_valid,
        'shear_valid': shear_valid,
        'aei_valid':   aei_valid,
    })

    meta = dict(r_H=rH, r_ISCO=rISCO, mm=mm, hr=hr, M=M, **params, **info)
    return df, meta