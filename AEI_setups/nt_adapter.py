"""
nt_adapter.py
=============
Adapter del modello Novikov-Thorne per aei_common.find_rossby.

ss_nt_boundaries.py calcola separatamente:
  - le frontiere r_AB, r_BC  (da Sigma0 → ṁ)
  - le formule di zona A/B/C per Σ(r)
  - i fattori relativistici A,B,C,D,E,Q

ma NON ha una funzione che restituisca direttamente  (B0, Sigma, c_s)
su un array di r con raccordo continuo tra zone.

Questo file aggiunge:

  nt_Sigma_disk(r, a, mdot, alpha, M)
      Σ(r) raccordata per continuità, esattamente come full_disk_SS
      ma usando le formule esatte NT (non la power-law raccordata).

  nt_B0_disk(r, norms, r_AB, r_BC)
      B₀(r) raccordata — riusa B0_disk da full_disk_SS perché B₀₀
      è un parametro libero non derivato dalle formule NT.

  nt_profiles(r_rg, a, B00, Sigma0, alpha, M, hr)
      Funzione-ponte completa: restituisce (B0, Sigma, c_s, zone)
      pronta per essere usata come disk_model in find_rossby.

  disk_model_NT(r_rg, a, B00, Sigma0, alpha_visc, hr, M)
      Wrapper diretto con la firma attesa da find_rossby.

──────────────────────────────────────────────────────────────────────
NOTA SUL RACCORDO DI Σ
──────────────────────────────────────────────────────────────────────
Le formule NT di Σ_A, Σ_B, Σ_C sono fisicamente valide solo nelle
rispettive zone. Alle frontiere NON sono necessariamente continue
(a differenza del raccordo power-law di full_disk_SS).

Strategia adottata qui: si usa la formula della zona corretta per
ogni r, con una piccola zona di smoothing di ±5% attorno a r_AB e r_BC
tramite media pesata logaritmica, per evitare salti netti.
Se preferisci la discontinuità fisica, usa nt_Sigma_disk(..., smooth=False).
"""

import numpy as np

import sys
sys.path.append("..")
from setup import r_isco, r_horizon, nu_phi, Rg_SUN, M_BH

from ss_nt_boundaries import (
    nt_boundaries,
    nt_mdot_from_Sigma0,
    nt_Sigma_A, nt_Sigma_B, nt_Sigma_C,
    nt_ABCDEQ,
)

from full_disk_SS import (
    compute_norms,
    B0_disk,
    sound_speed_disk,
    zone_index,
    ZONE_NAMES,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Σ(r) CON FORMULE NT  (raccordato per zona)
# ═══════════════════════════════════════════════════════════════════════════

def nt_Sigma_disk(r_rg, a, mdot, alpha, M=M_BH, r_AB=None, r_BC=None,
                  smooth=True, smooth_frac=0.05):
    """
    Σ(r) calcolata con le formule NT esatte, raccordata per zona.

    Parametri
    ----------
    r_rg       : array_like   raggi in r_g
    a          : float        spin
    mdot       : float        tasso di accrescimento adimensionale ṁ
    alpha      : float        viscosità α
    M          : float        massa BH  [M_sun]
    r_AB, r_BC : float        frontiere in r_g (se None vengono ricalcolate)
    smooth     : bool         se True applica smoothing nelle zone di transizione
    smooth_frac: float        larghezza relativa della zona di smoothing

    Restituisce
    -----------
    Sigma : ndarray  [g/cm²]
    """
    r_rg = np.asarray(r_rg, dtype=float)
    m    = float(M)

    if r_AB is None or r_BC is None:
        _, r_AB_calc, r_BC_calc = nt_boundaries(a, None, alpha, M)
        # se Sigma0 non è disponibile, usa i valori passati o fallback
        if r_AB is None: r_AB = r_AB_calc
        if r_BC is None: r_BC = r_BC_calc

    Sigma_A = nt_Sigma_A(r_rg, a, mdot, alpha)
    Sigma_B = nt_Sigma_B(r_rg, a, mdot, alpha)
    Sigma_C = nt_Sigma_C(r_rg, a, mdot, alpha, m)

    out = np.where(r_rg <= r_AB, Sigma_A,
          np.where(r_rg <= r_BC, Sigma_B, Sigma_C))

    if smooth:
        # zona di transizione A→B attorno a r_AB
        dr_AB = smooth_frac * r_AB
        mask_AB = np.abs(r_rg - r_AB) < dr_AB
        if np.any(mask_AB):
            w = (r_rg[mask_AB] - (r_AB - dr_AB)) / (2 * dr_AB)  # 0→1
            w = np.clip(w, 0, 1)
            out[mask_AB] = np.exp(
                (1 - w) * np.log(np.maximum(Sigma_A[mask_AB], 1e-300)) +
                w        * np.log(np.maximum(Sigma_B[mask_AB], 1e-300))
            )
        # zona di transizione B→C attorno a r_BC
        dr_BC = smooth_frac * r_BC
        mask_BC = np.abs(r_rg - r_BC) < dr_BC
        if np.any(mask_BC):
            w = (r_rg[mask_BC] - (r_BC - dr_BC)) / (2 * dr_BC)
            w = np.clip(w, 0, 1)
            out[mask_BC] = np.exp(
                (1 - w) * np.log(np.maximum(Sigma_B[mask_BC], 1e-300)) +
                w        * np.log(np.maximum(Sigma_C[mask_BC], 1e-300))
            )

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PROFILI COMPLETI  (B0, Sigma, c_s)
# ═══════════════════════════════════════════════════════════════════════════

def nt_profiles(r_rg, a, B00, Sigma0, alpha=0.1, M=M_BH, hr=0.05,
                smooth=True):
    """
    Profili fisici completi per il modello NT.

    B₀(r)  : power-law raccordata come in full_disk_SS
              (B₀₀ è parametro libero, non derivato da NT)
    Σ(r)   : formule esatte NT raccordate per zona
    c_s(r) : (H/r) × v_φ — identica a tutti i modelli

    Parametri
    ----------
    r_rg   : array_like   raggi in r_g
    a      : float        spin
    B00    : float        B₀ all'orizzonte  [G]
    Sigma0 : float        Σ all'ISCO  [g/cm²]
    alpha  : float        viscosità α
    M      : float        massa BH  [M_sun]
    hr     : float        aspect ratio H/r
    smooth : bool         smoothing di Σ alle frontiere

    Restituisce
    -----------
    B0    : ndarray  [G]
    Sigma : ndarray  [g/cm²]
    c_s   : ndarray  [cm/s]
    zone  : ndarray  di str  ('A', 'B', 'C')
    """
    r_rg = np.asarray(r_rg, dtype=float)

    # frontiere e ṁ
    mdot, r_AB, r_BC = nt_boundaries(a, Sigma0, alpha=alpha, M=M)

    # B₀: riusa il raccordo power-law di full_disk_SS
    norms = compute_norms(a, B00, Sigma0, r_AB, r_BC)
    B0    = B0_disk(r_rg, norms, r_AB, r_BC)

    # Σ: formule NT esatte
    Sigma = nt_Sigma_disk(r_rg, a, mdot, alpha, M,
                          r_AB=r_AB, r_BC=r_BC, smooth=smooth)

    # c_s: uguale a tutti i modelli
    c_s  = sound_speed_disk(r_rg, a, hr, M)

    # zona
    zi   = zone_index(r_rg, r_AB, r_BC)
    zone = np.array([ZONE_NAMES[i] for i in zi])

    return B0, Sigma, c_s, zone


# ═══════════════════════════════════════════════════════════════════════════
# 3.  ADAPTER PER find_rossby
# ═══════════════════════════════════════════════════════════════════════════

def disk_model_NT(r_rg, a, B00, Sigma0, alpha_visc=0.1, hr=0.05, M=M_BH):
    """
    Adapter con la firma attesa da find_rossby:

        B0, Sigma, c_s, zone = disk_model_NT(r_rg, **row_params)

    I parametri aggiuntivi (alpha_visc, hr, M) vanno fissati tramite
    lambda o functools.partial prima di passare a find_rossby.

    Parametri
    ----------
    r_rg      : array_like   raggi in r_g  (vettorizzato)
    a         : float        spin
    B00       : float        B₀ all'orizzonte  [G]
    Sigma0    : float        Σ all'ISCO  [g/cm²]
    alpha_visc: float        viscosità α
    hr        : float        aspect ratio H/r
    M         : float        massa BH  [M_sun]

    Restituisce
    -----------
    B0, Sigma, c_s : ndarray   profili fisici
    zone           : ndarray   etichette di zona ('A','B','C')
    """
    return nt_profiles(r_rg, a, B00, Sigma0,
                       alpha=alpha_visc, M=M, hr=hr)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  CELLE NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════════
"""
Celle da incollare nel notebook per usare find_rossby con il modello NT.
Sostituiscono find_rossby_matches_SS per questo modello.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELLA 1 — import
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from aei_common  import find_rossby
from nt_adapter  import disk_model_NT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELLA 2 — parametri (stessa struttura delle altre versioni)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

m          = 1
hr         = 0.05
alpha_visc = 0.1

# Griglia di parametri non-radiali
param_grid = {
    'a':      np.linspace(-0.9, 0.9, 19),
    'B00':    np.logspace(-3, 8, 24),      # G
    'Sigma0': np.logspace(3, 7, 20),       # g/cm²
}

# Griglia radiale (in r_g, escludi < ISCO automaticamente in find_rossby)
r_vec = np.geomspace(2, 1000, 150)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELLA 3 — run con tutti i check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df_NT = find_rossby(
    r_vec      = r_vec,
    param_grid = param_grid,
    disk_model = lambda r, **p: disk_model_NT(
                     r, **p,
                     alpha_visc=alpha_visc,
                     hr=hr,
                     M=M_BH,
                 ),
    m          = m,
    hr         = hr,
    M          = M_BH,
    check_k    = True,
    check_beta = True,
    check_shear= True,
)

print(f"Soluzioni trovate: {len(df_NT)}")
print(df_NT[['a','r','B00','Sigma0','k','beta','dQdr','zone']].head(10))

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELLA 4 — confronto SS vs NT sugli stessi parametri
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from full_disk_SS import (
    ss_boundaries, compute_norms,
    B0_disk, Sigma_disk, sound_speed_disk,
    zone_index, ZONE_NAMES,
)

def disk_model_SS(r_rg, a, B00, Sigma0, alpha_visc=0.1, hr=0.05, M=M_BH):
    r_AB, r_BC, _ = ss_boundaries(a, Sigma0, alpha=alpha_visc, M=M)
    norms         = compute_norms(a, B00, Sigma0, r_AB, r_BC)
    B0    = B0_disk(r_rg, norms, r_AB, r_BC)
    Sigma = Sigma_disk(r_rg, norms, r_AB, r_BC)
    c_s   = sound_speed_disk(r_rg, a, hr, M)
    zi    = zone_index(r_rg, r_AB, r_BC)
    zone  = np.array([ZONE_NAMES[i] for i in zi])
    return B0, Sigma, c_s, zone

df_SS = find_rossby(
    r_vec, param_grid,
    disk_model=lambda r, **p: disk_model_SS(r, **p, alpha_visc=alpha_visc,
                                             hr=hr, M=M_BH),
    m=m, hr=hr, M=M_BH,
    check_k=True, check_beta=True, check_shear=True,
)

print(f"SS: {len(df_SS)} soluzioni   |   NT: {len(df_NT)} soluzioni")

# overlay: stessa colonna 'zone' in entrambi
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, df, label in zip(axes, [df_SS, df_NT], ['SS power-law', 'NT esatto']):
    sc = ax.scatter(df['r'], df['k'], c=np.log10(df['B00']),
                    cmap='plasma', s=10, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('r  [r_g]')
    ax.set_ylabel('k  [adim]')
    ax.set_title(label)
    plt.colorbar(sc, ax=ax, label='log B₀₀  [G]')
plt.tight_layout()
plt.show()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELLA 5 — diagnostica frontiere NT per una combinazione specifica
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from ss_nt_boundaries import nt_print_boundaries

nt_print_boundaries(a=0.5, Sigma0=1e5, alpha=alpha_visc)

# profilo radiale per un singolo set di parametri
from nt_adapter import nt_profiles
import numpy as np

r_test  = np.geomspace(2, 500, 200)
B0_t, S_t, cs_t, zone_t = nt_profiles(r_test, a=0.5, B00=1e7,
                                        Sigma0=1e5, alpha=0.1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
zone_colors = {'A':'#f97316', 'B':'#3b82f6', 'C':'#22c55e'}
for zn, col in zone_colors.items():
    mask = zone_t == zn
    if not np.any(mask): continue
    axes[0].loglog(r_test[mask], B0_t[mask],  '.', color=col, label=zn)
    axes[1].loglog(r_test[mask], S_t[mask],   '.', color=col)
    axes[2].loglog(r_test[mask], cs_t[mask],  '.', color=col)
for ax, ylabel in zip(axes, ['B₀ [G]', 'Σ [g/cm²]', 'c_s [cm/s]']):
    ax.set_xlabel('r [r_g]');  ax.set_ylabel(ylabel)
axes[0].legend()
plt.suptitle('Profili NT — a=0.5, B₀₀=1e7, Σ₀=1e5')
plt.tight_layout()
plt.show()
"""
