"""
align_timescale.py

Tempo di allineamento viscoso del flusso interno precessante (Foucart &
Lai 2014, Eq. 7 di Motta et al. 2017), condizione di disallineamento
necessaria (assieme a t_wave) per la precessione rigida coerente del
PIF.

Implementazione vettoriale: un integrale cumulativo (G_phi) piu' due
integrali definiti (numeratore/denominatore di gamma), nessun ciclo
Python esplicito sui punti della griglia radiale.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from setup import nu_phi, nu_theta
from disk_profiles import (Sigma_torque_zero, c_s_powerlaw,
                            P_TORQUE, HR_ISCO_DEFAULT, Q_SOUND)
from nu_solid_v2 import nu_solid_vect_p


def t_align_vect(a, r_in, r_out, M, alpha, p=P_TORQUE,
                  HR_isco=HR_ISCO_DEFAULT, q=Q_SOUND, n_rad=1000):
    """
    Tempo di allineamento t_align = 1/gamma (Eq. 7 di Motta et al. 2017).

    a, r_in, r_out, M, alpha : array_like, tutti broadcastabili tra loro.
    Ritorna t_align [s], stessa shape del broadcast.
    """
    shape = np.broadcast(
        np.asarray(a, dtype=float), np.asarray(r_in, dtype=float),
        np.asarray(r_out, dtype=float), np.asarray(M, dtype=float),
        np.asarray(alpha, dtype=float)).shape

    a = np.broadcast_to(np.asarray(a, dtype=float), shape)
    r_in = np.broadcast_to(np.asarray(r_in, dtype=float), shape)
    r_out = np.broadcast_to(np.asarray(r_out, dtype=float), shape)
    M = np.broadcast_to(np.asarray(M, dtype=float), shape)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), shape)

    # xi parte da un epsilon>0, non da 0: a R=r_in, Sigma_torque_zero=0
    # esattamente (per costruzione, Eq. 3 di Motta), il che produrrebbe
    # una divisione 0/0 nell'integrando del numeratore (G_phi^2/Sigma).
    # G_phi(r_in)=0 e Sigma(r_in)=0 allo stesso ordine (~sqrt(R-r_in)),
    # quindi il rapporto e' in realta' finito nel limite R->r_in+, ma
    # va evitato il punto esatto per non generare un NaN numerico.
    eps = 1e-6
    xi = np.linspace(eps, 1.0, n_rad)
    xi = xi.reshape((1,) * a.ndim + (n_rad,))
    R = r_in[..., None] + xi * (r_out - r_in)[..., None]

    a_b = a[..., None]
    M_b = M[..., None]
    alpha_b = alpha[..., None]
    r_in_b = r_in[..., None]

    Omega_K = 2 * np.pi * nu_phi(R, a_b, M_b)     # rad/s
    Omega_z = 2 * np.pi * nu_theta(R, a_b, M_b)   # rad/s (~Omega_perp)

    Omega_p = 2 * np.pi * nu_solid_vect_p(a, r_in, r_out, M, p=p)
    Omega_p_b = Omega_p[..., None]

    Z = (Omega_K**2 - Omega_z**2) / (2 * Omega_K**2)

    Sigma = Sigma_torque_zero(R, r_in_b, p)
    c_s = c_s_powerlaw(R, a_b, HR_isco, q)

    integrand_G = Sigma * R**3 * Omega_K * (Omega_p_b - Z * Omega_K)
    G_phi = cumulative_trapezoid(integrand_G, R, axis=-1, initial=0.0)

    num_integrand = 4 * alpha_b * G_phi**2 / (Sigma * c_s**2 * R**3)
    den_integrand = Sigma * R**3 * Omega_K

    numerator = np.trapezoid(num_integrand, R, axis=-1)
    denominator = np.trapezoid(den_integrand, R, axis=-1)

    gamma_hat = numerator / denominator

    # CORREZIONE DIMENSIONALE: l'integrale e' stato svolto sulla
    # variabile adimensionale x=R/Rg (dR -> dx, R^3 -> x^3), ma la
    # formula fisica di Foucart-Lai (Eq. 7 Motta) e' definita su R
    # fisico. La sostituzione R=Rg*x propaga un fattore Rg^2 netto su
    # gamma (derivazione: G_phi~Rg^4, Numeratore~Rg^6, Denominatore~Rg^4,
    # quindi gamma_phys = Rg^2 * gamma_hat). Rg in cm.
    from setup import Rg_SUN
    Rg_cm = Rg_SUN * M  # cm, per ciascun punto della griglia (a,r_in,r_out,M)
    gamma = gamma_hat * Rg_cm**2

    return 1.0 / gamma