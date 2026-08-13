"""
nu_solid_v2.py

Frequenza di precessione PIF con il profilo di densita' a torque nullo
(disk_profiles.Sigma_torque_zero, p=3/5 fisso), al posto del profilo
Sigma ~ R^-zeta liberamente scansionato in pif_v2.py.

Scaling esatta in M
--------------------
nu_LT(r,a,M) = g_LT(r,a)/M e nu_phi(r,a,M) = g_phi(r,a)/M (esatto,
setup.py). Nel peso w = Sigma(R)*R^3*nu_phi(R,a,M) il fattore 1/M
compare linearmente e si cancella nel rapporto numeratore/denominatore
di nu_solid, perche' Sigma(R) e R non dipendono da M. Quindi

    nu_solid(a, r_in, r_out, p, M) = g_solid(a, r_in, r_out, p) / M

con g_solid puramente geometrico, calcolabile una sola volta con M=1.
Permette l'inversione analitica di M dato un nu0 target, senza
bisezione numerica (stessa logica di param_space.mass_envelope_analytic
per RPM).
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from setup import r_isco, g_phi, g_LT
from disk_profiles import Sigma_torque_zero, P_TORQUE


def _nu_solid_geometric(a, r_in, r_out, p=P_TORQUE, n_rad=2000):
    """g_solid(a, r_in, r_out, p) = nu_solid_vect(..., M=1)."""
    a = np.atleast_1d(a).astype(float)
    r_in = np.atleast_1d(r_in).astype(float)
    r_out = np.atleast_1d(r_out).astype(float)

    x = np.linspace(0.0, 1.0, n_rad)
    x = x.reshape((1,) * a.ndim + (n_rad,))

    R = r_in[..., None] + x * (r_out - r_in)[..., None]
    a_b = a[..., None]

    Sigma = Sigma_torque_zero(R, r_in[..., None], p)
    weight = Sigma * R**3 * g_phi(R, a_b)   # g_phi = M*nu_phi, M=1 qui

    num = np.trapezoid(g_LT(R, a_b) * weight, x, axis=-1)
    den = np.trapezoid(weight, x, axis=-1)
    return num / den


def nu_solid_vect_p(a, r_in, r_out, M, p=P_TORQUE, n_rad=2000):
    """Frequenza PIF fisica: nu_solid(a, r_in, r_out, p, M) = g_solid/M."""
    g_solid = _nu_solid_geometric(a, r_in, r_out, p, n_rad)
    return g_solid / np.asarray(M, dtype=float)


def mass_from_nu_solid(a, r_in, r_out, nu_target, p=P_TORQUE, n_rad=2000):
    """
    Inversione analitica: M = g_solid(a, r_in, r_out, p) / nu_target.
    Nessuna bisezione.
    """
    g_solid = _nu_solid_geometric(a, r_in, r_out, p, n_rad)
    return g_solid / np.asarray(nu_target, dtype=float)
