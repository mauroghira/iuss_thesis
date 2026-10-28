"""
pif_core.py

With p, q fixed, the three PIF quantities depend on M, alpha and HR_isco
through exact, factorisable laws:

    nu_solid = g / M                                       [g     in Hz Msun]
    t_wave   = M * tau_w * (HR_ref / HR_isco)              [tau_w in s / Msun]
    t_align  = M * tau_a * (HR_isco / HR_ref)^2 * (ALPHA / alpha)
                                                           [tau_a in s / Msun]

(t_align is inversely proportional to alpha, t_wave does not depend on it.)
g, tau_w, tau_a depend ONLY on (a, r_in, r_out, p, q) and are computed once
at M = 1, alpha = ALPHA, HR_isco = HR_ref; tau_w = t_wave / M and
tau_a = t_align / M are therefore directly the timescales per unit mass at
the reference alpha and HR. A source with observed frequency nu_0 and mass M
enters only through P = M * nu_0:

    precession : g(r_out)     = P
    t_wave     : tau_w(r_out) = 1 / P
    t_align    : tau_a(r_out) = 1 / P

(at alpha = ALPHA, HR_isco = HR_ref; for another alpha or HR_isco use
PIFGeometric.physical).

Units: r = R / R_g is dimensionless, M is in solar masses, times in seconds.

Dependencies from setup.py (they stay there because other models use them):
C, Rg_SUN, r_isco, g_phi, g_theta, g_LT.

References: Motta et al. 2017 (arXiv:1709.02608), Foucart & Lai 2014.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from setup import C, Rg_SUN, r_isco, g_phi, g_theta, g_LT

_trapz = getattr(np, 'trapezoid', None) or np.trapz

# --------------------------------------------------------------------------
# Disc prescriptions (Motta et al. 2017)
# --------------------------------------------------------------------------
P_TORQUE = 3.0 / 5.0      # Sigma ~ R^-p (1 - sqrt(r_in/R))^p, zero torque
Q_SOUND = 3.0 / 2.0       # c_s ~ R^-q, radiation-pressure dominated disc
HR_ISCO_DEFAULT = 0.1     # (H/R) at the ISCO, fixed by hand (not derived)
ALPHA = 0.01              # Shakura-Sunyaev viscosity parameter (fixed here)

# Numerical parameters
N_RAD = 2000              # radial nodes of the integrals
EPS = 1e-6                # xi starts at EPS: Sigma(r_in) = 0 would give 0/0 in G^2/Sigma
_MAX_ELEMS = 4_000_000    # max elements of a (n_points x n_rad) array per chunk


def sigma_torque_zero(R, r_in, p=P_TORQUE):
    """Sigma(R) = R^-p (1 - sqrt(r_in/R))^p, with Sigma0 = 1 (neither nu_solid
    nor t_align depend on the absolute normalisation). Requires R >= r_in."""
    R = np.asarray(R, dtype=float)
    ratio = np.clip(r_in / R, 0.0, 1.0)
    return R**(-p) * (1.0 - np.sqrt(ratio))**p


# --------------------------------------------------------------------------
# Geometric result (M = 1, alpha = ALPHA, HR_isco = HR_ref)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PIFGeometric:
    """
    Geometric curves, all with the same shape (broadcast of a, r_in, r_out).
    g     [Hz Msun]  : nu_solid * M
    tau_w [s / Msun] : t_wave / M    (at HR_isco = HR_ref)
    tau_a [s / Msun] : t_align / M   (at alpha = alpha_ref, HR_isco = HR_ref)
    NaN where r_out <= r_in.
    """
    g: np.ndarray
    tau_w: np.ndarray
    tau_a: np.ndarray
    HR_ref: float = HR_ISCO_DEFAULT
    alpha_ref: float = ALPHA

    def physical(self, M, alpha=None, HR_isco=None):
        """
        Returns (nu_solid [Hz], t_wave [s], t_align [s]) for M [Msun] and,
        optionally, a different alpha or HR_isco (defaults: the reference
        values of the curves). M, alpha, HR_isco must broadcast with the
        curve shape (add axes with [..., None] if needed).

        t_wave ~ M / HR_isco;  t_align ~ M * HR_isco^2 / alpha.
        """
        HR = self.HR_ref if HR_isco is None else HR_isco
        al = self.alpha_ref if alpha is None else alpha
        M = np.asarray(M, dtype=float)
        nu = self.g / M
        t_wave = M * self.tau_w * (self.HR_ref / HR)
        t_align = M * self.tau_a * (HR / self.HR_ref)**2 * (self.alpha_ref / al)
        return nu, t_wave, t_align


# --------------------------------------------------------------------------
# Numerical kernels (one chunk of flattened points)
# --------------------------------------------------------------------------
def _radial_setup(a, r_in, r_out, p, n_rad):
    """
    Radial grid R = r_in + xi (r_out - r_in), xi in [EPS, 1], and the
    quantities shared by nu_solid and t_align. All inputs are 1D (n,);
    outputs are (n, n_rad) arrays (a2 is (n, 1)).
    """
    xi = np.linspace(EPS, 1.0, n_rad)
    R = r_in[:, None] + xi[None, :] * (r_out - r_in)[:, None]
    a2 = a[:, None]
    gphi = g_phi(R, a2)                          # Omega_K / (2 pi) at M = 1
    Sigma = sigma_torque_zero(R, r_in[:, None], p)
    w = Sigma * R**3 * gphi                      # common weight
    return R, a2, gphi, Sigma, w


def _kernel_g(a, r_in, r_out, p, n_rad):
    """nu_solid * M only (cheap): returns a 1-tuple (g,)."""
    R, a2, _, _, w = _radial_setup(a, r_in, r_out, p, n_rad)
    g = _trapz(g_LT(R, a2) * w, R, axis=-1) / _trapz(w, R, axis=-1)
    return (g,)


def _kernel_full(a, r_in, r_out, p, q, HR_ref, alpha, n_rad):
    """
    (g, tau_w, tau_a) for 1D arrays a, r_in, r_out with r_out > r_in, at
    M = 1 and the given alpha, HR_ref.
    A single radial grid is shared by nu_solid and t_align: Omega_p is not
    recomputed elsewhere.
    """
    R, a2, gphi, Sigma, w = _radial_setup(a, r_in, r_out, p, n_rad)

    # --- nu_solid * M ---------------------------------------------------
    gs = _trapz(g_LT(R, a2) * w, R, axis=-1) / _trapz(w, R, axis=-1)

    # --- t_wave / M (Eq. 6 of Motta+17, closed form) --------------------
    r_isco_a = r_isco(a)
    tau_w = (2.0 * Rg_SUN / C / HR_ref * r_isco_a**(0.5 - q)
             * (r_out**(1.0 + q) - r_in**(1.0 + q)) / (1.0 + q))

    # --- t_align / M (Eq. 7 of Motta+17, Foucart & Lai 2014) ------------
    # Z = (Omega_K^2 - Omega_z^2) / (2 Omega_K^2) = (1 - (g_theta/g_phi)^2) / 2
    gth = g_theta(R, a2)
    Z = 0.5 * (1.0 - (gth / gphi)**2)
    two_pi = 2.0 * np.pi
    # Sigma R^3 Omega_K (Omega_p - Z Omega_K) = (2 pi)^2 w (g_solid - Z gphi)
    integrand_G = two_pi**2 * w * (gs[:, None] - Z * gphi)
    G = cumulative_trapezoid(integrand_G, R, axis=-1, initial=0.0)

    HR0 = HR_ref * r_isco_a**(q - 0.5)
    cs2 = (C * HR0[:, None])**2 * R**(-2.0 * q)
    num = _trapz(4.0 * alpha * G**2 / (Sigma * cs2 * R**3), R, axis=-1)
    den = _trapz(two_pi * w, R, axis=-1)                          # Sigma R^3 Omega_K
    # R is in units of r_g: gamma_phys = gamma_hat * Rg^2, Rg = Rg_SUN at M = 1
    gamma = (num / den) * Rg_SUN**2
    return gs, tau_w, 1.0 / gamma


def _evaluate(kernel, n_out, a, r_out, r_in, n_rad):
    """
    Broadcast (a, r_out, r_in), evaluate `kernel` on chunks of the flattened
    valid points (r_out > r_in) and return n_out arrays of the broadcast
    shape, NaN where r_out <= r_in. Memory use is bounded by _MAX_ELEMS
    (chunk size = _MAX_ELEMS // n_rad points).

    kernel(a, r_in, r_out) -> tuple of n_out 1D arrays.
    """
    a = np.asarray(a, dtype=float)
    r_out = np.asarray(r_out, dtype=float)
    r_in = r_isco(a) if r_in is None else np.asarray(r_in, dtype=float)

    shape = np.broadcast_shapes(a.shape, r_in.shape, r_out.shape)
    a_f = np.broadcast_to(a, shape).ravel()
    ri_f = np.broadcast_to(r_in, shape).ravel()
    ro_f = np.broadcast_to(r_out, shape).ravel()

    outs = [np.full(a_f.size, np.nan) for _ in range(n_out)]
    idx = np.flatnonzero(ro_f > ri_f)
    step = max(1, _MAX_ELEMS // n_rad)
    for s in range(0, idx.size, step):
        sl = idx[s:s + step]
        for out, val in zip(outs, kernel(a_f[sl], ri_f[sl], ro_f[sl])):
            out[sl] = val
    return [o.reshape(shape) for o in outs]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def pif_geometric(a, r_out, r_in=None, p=P_TORQUE, q=Q_SOUND,
                  HR_ref=HR_ISCO_DEFAULT, alpha=ALPHA, n_rad=N_RAD):
    """
    (g, tau_w, tau_a) at M = 1, alpha = ALPHA, HR_isco = HR_ref.

    a, r_out, r_in : array_like, mutually broadcastable (r_in = r_isco(a) if
    None). Evaluation is chunked, so arbitrarily large (a, r_out) grids fit
    in memory. Points with r_out <= r_in are NaN.
    """
    g, tau_w, tau_a = _evaluate(
        lambda a_, ri_, ro_: _kernel_full(a_, ri_, ro_, p, q, HR_ref, alpha, n_rad),
        3, a, r_out, r_in, n_rad)
    return PIFGeometric(g, tau_w, tau_a, HR_ref, alpha)


def g_solid(a, r_out, r_in=None, p=P_TORQUE, n_rad=N_RAD):
    """g = nu_solid * M [Hz Msun] only; much cheaper than pif_geometric
    because t_align (cumulative integral) is not computed."""
    (g,) = _evaluate(
        lambda a_, ri_, ro_: _kernel_g(a_, ri_, ro_, p, n_rad),
        1, a, r_out, r_in, n_rad)
    return g


def mass_from_nu_solid(a, r_out, nu_target, r_in=None, p=P_TORQUE,
                       n_rad=N_RAD):
    """Exact inversion: M = g(a, r_in, r_out, p) / nu_target."""
    return g_solid(a, r_out, r_in, p, n_rad) / np.asarray(nu_target, dtype=float)