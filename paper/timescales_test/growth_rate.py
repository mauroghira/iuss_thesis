"""
growth_rate.py

Tasso di crescita/smorzamento viscoso G(r) di Kato (2001), Eq. (73),
valido SOLO per il modo fondamentale assisimmetrico (m=0, n=0,
p-mode), generalizzato alla metrica di Kerr.

Derivazione della semplificazione usata qui
--------------------------------------------
k_r(r) fissato risolvendo la relazione di dispersione locale del
p-mode (Eq. 34/72 di Kato, n=0): omega^2 = kappa^2 + c_T^2 k_r^2, cioe'

    k_r^2(r) = (omega^2 - kappa(r)^2) / c_T(r)^2

Sostituendo in Eq. (73):
    (eta0/rho0)*k_r^2 = alpha*(omega^2-kappa^2)/Omega
    kappa^2 + c_T^2 k_r^2 = omega^2   (esatto)

c_T si cancella interamente. Risultato:

    G(r) = -alpha*(omega^2-kappa(r)^2)/Omega(r)
           *[ Omega(r)^2/omega^2*(dlnOmega/dlnr(r) + kappa(r)^2/(2 Omega(r)^2))
              + 2/3 ]

con A=1 (perturbazioni isoterme, Eq. 69 di Kato con A=1). Valida solo
per omega^2 > kappa(r)^2 (finestra di propagazione del p-mode, Eq. 56).
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from setup import nu_r, nu_phi, r_isco


def dlnOmega_dlnr(r, a):
    """
    d ln(Omega_phi)/d ln(r), analitica, indipendente da M.
    g_phi(r,a) = cost/(r^1.5+a) => d ln g_phi/d ln r = -1.5*r^1.5/(r^1.5+a)
    Limite kepleriano newtoniano (a=0): -1.5 (Kato, Sez. 6.1).
    """
    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)
    return -1.5 * r**1.5 / (r**1.5 + a)


def growth_rate_p_mode(r, a, M, nu0, alpha, A=1.0):
    """
    G(r) per il p-mode fondamentale (m=0, n=0). NaN dove omega^2<=kappa(r)^2
    (fuori dalla finestra di propagazione).
    """
    r = np.asarray(r, dtype=float)
    kappa = 2 * np.pi * nu_r(r, a, M)      # rad/s
    Omega = 2 * np.pi * nu_phi(r, a, M)    # rad/s
    omega = 2 * np.pi * np.asarray(nu0, dtype=float)  # rad/s

    valid = omega**2 > kappa**2
    dlnOm = dlnOmega_dlnr(r, a)

    bracket = (Omega**2 / omega**2) * (A * dlnOm + kappa**2 / (2 * Omega**2)) + 2.0 / 3.0
    G = -alpha * (omega**2 - kappa**2) / Omega * bracket

    return np.where(valid, G, np.nan)


def find_p_mode_outer_boundary(a, M, nu0, r_scan_max=50.0, n_scan=8000):
    """
    r1: primo raggio (da ISCO verso l'esterno) dove kappa(r) = omega,
    bordo esterno della regione di trapping del p-mode fondamentale
    (Kato, Fig. 6, Sez. 5.1: regione intrappolata = [r_isco, r1]).
    Scalari (root-finding singolo).
    """
    a = float(a); M = float(M); nu0 = float(nu0)
    r_in = float(r_isco(a))

    r_scan = np.geomspace(r_in * 1.0001, r_scan_max, n_scan)
    kappa_scan = nu_r(r_scan, a, M)
    diff = nu0 - kappa_scan

    sign_changes = np.where(np.diff(np.sign(diff)) < 0)[0]
    if len(sign_changes) == 0:
        return np.nan

    i = sign_changes[0]
    denom = diff[i + 1] - diff[i]
    if denom == 0:
        return float(r_scan[i])
    r1 = r_scan[i] - diff[i] * (r_scan[i + 1] - r_scan[i]) / denom
    return float(r1)
