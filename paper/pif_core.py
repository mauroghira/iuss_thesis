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

==========================================================================
TWO WAYS OF OBTAINING (g, tau_a)            parameter `method`
==========================================================================
method='motta' (default): closed-form, first-order solution of Foucart & Lai
    2014 (FL14) as used by Motta+17: Omega_p from an integral average and
    t_align = 1/gamma from FL14 Eq. 35 (Motta Eq. 7). Fast (vectorised).
    Valid only for small warps, small twist gradient, gamma << alpha Omega
    and |Omega_z^2 - Omega^2| <~ (H/r) Omega^2 (FL14, text after Eq. 7 and
    Eq. 29). The torque closure is selected by `closure`:
      'lt'      (Motta+17 literally) Omega_p = <Omega_LT>_w inserted in
                G = int w (Omega_p - Z Omega_K);  G(R_out) = (1/2) int
                Sigma R^3 Omega_LT^2 dR > 0 (no closure of the torque).
      'z'       Omega_p = <Z Omega_K>_w inside G only (G(R_out) = 0); the
                observable g stays <Omega_LT>.
      'lt_exact' local rate Omega_LT = Omega_phi - Omega_z everywhere
                (average and G): G(R_out) = 0 and g = <Omega_LT>.
      'fl'      FL14 literally: Omega_p = <Z Omega_K>_w both in G and as
                the observable g (FL14 Eq. 30).
    g is <Omega_LT> for 'lt', 'z', 'lt_exact' and <Z Omega_K> for 'fl'.
method='eig': full linear eigenmode of the bending-wave equations (FL14
    Eqs. 17-18, W_- mode, boundary conditions G(r_in) = G(r_out) = 0), with
    Kerr Omega, Omega_z, kappa (the same equations that Lubow, Ogilvie &
    Pringle 2002 use for Kerr). Solves  omega = omega_p - i gamma  without
    the first-order assumptions; g = |Re omega|/2 pi and tau_a = 1/gamma come
    from the SAME mode. The mode selected is the flattest one (angular-
    momentum-weighted tilt closest to rigid), i.e. FL14's lowest-order mode.
    CAVEAT: the flattest mode coincides with the least-damped one when a nearly
    rigid mode exists (weak field, small r_out), but not always (e.g. a = 0.9,
    r_out = 30, flatness ~0.4): the PIFGeometric.extra['flatness'] array should
    be inspected before trusting eigenmode values.
    `rate` sets the local rates entering the equations:
      'lt': Z Omega -> Omega_phi - Omega_z,  K Omega -> Omega_phi - kappa
      'z' : Z Omega = (Omega^2 - Omega_z^2)/(2 Omega),
            K Omega = (Omega^2 - kappa^2)/(2 Omega)   (FL14 Eqs. 15-16)
    NB: the eigenproblem depends on (HR_isco, alpha) NOT only through the
    simple scalings above: the curves are valid only for the (HR_ref, alpha)
    used to compute them, and PIFGeometric.physical refuses other values.
    t_wave stays the closed form of Motta Eq. 6 in both methods.
    The first-order counterparts of the eigenmodes are closure='lt_exact'
    (rate='lt') and closure='fl' (rate='z'): they coincide with the
    eigenmode in the weak-field / small-warp limit.

M scaling holds in both methods (omega ~ 1/M), so validity still depends on
(M, nu_0) only through P = M nu_0.

References: Motta et al. 2017 (arXiv:1709.02608), Foucart & Lai 2014
(arXiv:1406.3331), Lubow, Ogilvie & Pringle 2002 (astro-ph/0208206).
"""

from __future__ import annotations

import os
import sys
import tqdm
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid

from setup import C, Rg_SUN, r_isco, g_phi, g_theta, g_r, g_LT

_trapz = getattr(np, 'trapezoid', None) or np.trapz

# --------------------------------------------------------------------------
# Disc prescriptions (Motta et al. 2017)
# --------------------------------------------------------------------------
P_TORQUE = 3.0 / 5.0      # Sigma ~ R^-p (1 - sqrt(r_in/R))^p, zero torque
Q_SOUND = 3.0 / 2.0       # c_s ~ R^-q, radiation-pressure dominated disc
HR_ISCO_DEFAULT = 0.1     # (H/R) at the ISCO, fixed by hand (not derived)
ALPHA = 0.01              # Shakura-Sunyaev viscosity parameter (fixed here)
CLOSURES = ('lt', 'z', 'lt_exact', 'fl')   # first-order closures (method='motta')
CLOSURE_DEFAULT = 'lt_exact'    # 'lt_exact' = literal Motta+17 (Omega_p from exact Omega_LT)
METHODS = ('motta', 'eig')
METHOD_DEFAULT = 'motta'
RATES = ('lt', 'z')       # local precession / periastron rates (method='eig')
RATE_DEFAULT = 'lt'
N_EIG = 120               # radial cells of the eigenproblem (matrix 2N-1)

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
    method: str = METHOD_DEFAULT
    # optional per-point diagnostics, same shape as g:
    #   method='motta', diagnostics=True: 'dl_theta', 'psi_max' (per unit
    #       sin(beta_out)), 'gamma_over_alpha_omega' (gamma / alpha Omega(r_out))
    #   method='eig': 'flatness' in [0, 1] of the selected mode (1 = rigid)
    extra: dict = field(default_factory=dict)

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
        if self.method == 'eig' and not (np.all(np.asarray(HR) == self.HR_ref)
                                         and np.all(np.asarray(al) == self.alpha_ref)):
            raise ValueError("method='eig' curves are not rescalable in alpha or "
                             "HR_isco: recompute them with pif_geometric(alpha=..., "
                             "HR_ref=...)")
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


def _kernel_full(a, r_in, r_out, p, q, HR_ref, alpha, n_rad,
                 closure=CLOSURE_DEFAULT, diagnostics=False):
    """
    (g, tau_w, tau_a) for 1D arrays a, r_in, r_out with r_out > r_in, at
    M = 1 and the given alpha, HR_ref.
    A single radial grid is shared by nu_solid and t_align: Omega_p is not
    recomputed elsewhere.

    closure: see the module docstring ('lt', 'z', 'lt_exact', 'fl').
    diagnostics=True appends three arrays (FL14 Eqs. 28, 34, 35):
      dl_theta  = |l_theta(r_in) - l_theta(r_out)| per unit sin(beta_out)
      psi_max   = max_r |d l / d ln r| per unit sin(beta_out)
      gamma / (alpha Omega(r_out))
    The first-order solution requires all three (and psi <~ H/r) << 1.
    """
    R, a2, gphi, Sigma, w = _radial_setup(a, r_in, r_out, p, n_rad)

    # --- nu_solid * M ---------------------------------------------------
    # Omega_LT = Omega_phi - Omega_z is SIGNED: negative for a < 0 (g_theta >
    # g_phi, retrograde precession). The torque integral below needs the signed
    # Omega_solid, so that Omega_solid - Z Omega_K cancels also for a < 0.
    # The observable is nu_solid = |Omega_solid| / 2 pi, i.e. gs = |gs_signed|
    # (g_phi - g_theta has the sign of a at every R >= r_isco >= 1).
    gth = g_theta(R, a2)
    w_tot = _trapz(w, R, axis=-1)
    gs_signed = _trapz((gphi - gth) * w, R, axis=-1) / w_tot
    gs = np.abs(gs_signed)

    # --- t_wave / M (Eq. 6 of Motta+17, closed form) --------------------
    r_isco_a = r_isco(a)
    tau_w = (2.0 * Rg_SUN / C / HR_ref * r_isco_a**(0.5 - q)
             * (r_out**(1.0 + q) - r_in**(1.0 + q)) / (1.0 + q))

    # --- t_align / M (Eq. 7 of Motta+17, Foucart & Lai 2014) ------------
    # Z = (Omega_K^2 - Omega_z^2) / (2 Omega_K^2) = (1 - (g_theta/g_phi)^2) / 2
    Z = 0.5 * (1.0 - (gth / gphi)**2)
    two_pi = 2.0 * np.pi
    loc = Z * gphi                                        # Z Omega_K / 2 pi
    if closure == 'lt':
        g_p = gs_signed                                   # <Omega_LT>_w
    elif closure == 'z':
        g_p = _trapz(Z * gphi * w, R, axis=-1) / w_tot    # <Z Omega_K>_w
    elif closure == 'lt_exact':
        g_p = gs_signed                                   # <Omega_LT>_w
        loc = gphi - gth                                  # Omega_LT everywhere
    elif closure == 'fl':
        g_p = _trapz(Z * gphi * w, R, axis=-1) / w_tot    # <Z Omega_K>_w
        gs = np.abs(g_p)                                  # observable too
    else:
        raise ValueError(f"closure must be one of {CLOSURES}, got {closure!r}")
    # Sigma R^3 Omega_K (Omega_p - loc) = (2 pi)^2 w (g_p - loc)
    integrand_G = two_pi**2 * w * (g_p[:, None] - loc)
    G = cumulative_trapezoid(integrand_G, R, axis=-1, initial=0.0)

    HR0 = HR_ref * r_isco_a**(q - 0.5)
    cs2 = (C * HR0[:, None])**2 * R**(-2.0 * q)
    num = _trapz(4.0 * alpha * G**2 / (Sigma * cs2 * R**3), R, axis=-1)
    den = _trapz(two_pi * w, R, axis=-1)                          # Sigma R^3 Omega_K
    # R is in units of r_g: gamma_phys = gamma_hat * Rg^2, Rg = Rg_SUN at M = 1
    gamma = (num / den) * Rg_SUN**2
    if not diagnostics:
        return gs, tau_w, 1.0 / gamma
    # periastron term K Omega of FL14 Eq. 15 (same convention as the local rate)
    gr_ = g_r(R, a2)
    if closure == 'lt_exact':
        K_loc = gphi - gr_
    else:
        K_loc = 0.5 * (1.0 - (gr_ / gphi)**2) * gphi
    Om = two_pi * gphi
    dd = two_pi * (K_loc - g_p[:, None])                  # K Omega - Omega_p
    pref = 4.0 * Rg_SUN**2 / (Sigma * cs2)                # 4 / (Sigma c_s^2) in code units
    dl_dr = pref * G * dd / (R**3 * Om)                   # FL14 Eq. 28
    dl_theta = np.abs(_trapz(dl_dr, R, axis=-1))
    psi = pref * np.abs(G) / (R**2 * Om) * np.sqrt(
        (alpha * Om)**2 + dd**2)                          # FL14 Eq. 34
    gam_ratio = gamma / (alpha * Om[:, -1])
    return gs, tau_w, 1.0 / gamma, dl_theta, np.max(psi, axis=-1), gam_ratio


# --------------------------------------------------------------------------
# Full linear eigenmode (FL14 Eqs. 17-18, W_- mode)
# --------------------------------------------------------------------------
def warp_eigenmode(r_in, r_out, coef, alpha, n=N_EIG, return_all=False):
    """
    Lowest-order (flattest) eigenmode omega = omega_p - i gamma of the linear
    bending-wave equations (W = l_x - i l_y, G = G_x - i G_y), with Sigma_0 = 1:

        dG/dr = -i Sigma r^3 Omega (omega - Z Omega) W
        dW/dr = [4 / (Sigma r^3 c_s^2 Omega)] (-i omega + alpha Omega + i K Omega) G
        G(r_in) = G(r_out) = 0.

    Both equations are linear in omega, so after a staggered finite-
    difference discretisation (W at the N cell centres, G at the N-1 interior
    nodes, geometric grid r_in -> r_out) omega is an ordinary eigenvalue of a
    dense (2N-1)x(2N-1) complex matrix:

        omega W_j = (Z Omega)_j W_j + i/(s1_j dr_j) (G_{j+1} - G_j),   s1 = Sigma r^3 Omega
        omega G_i = i/(a_i (rc_i - rc_{i-1})) (W_i - W_{i-1}) + (K Omega - i alpha Omega)_i G_i,
        a = 4 / (Sigma r^3 c_s^2 Omega).

    coef(r) -> (Sigma, Omega, cs2, ZOmega, KOmega) for an array r, all in one
    consistent unit system (Omega, rates in rad/s; cs2 in r-units^2/s^2).
    The selected mode maximises the flatness
        F = |sum s1 W dr|^2 / (sum s1 dr * sum s1 |W|^2 dr)  in [0, 1],
    i.e. the angular-momentum-weighted tilt closest to rigid precession.
    Returns (omega, F) or, with return_all=True, all eigenvalues and F values.
    """
    N = int(n)
    x = np.linspace(0.0, 1.0, N + 1)
    r = r_in * (r_out / r_in) ** x
    rc = 0.5 * (r[:-1] + r[1:])
    dr = np.diff(r)
    Sc, Oc, _, zc, _ = coef(rc)
    rn = r[1:-1]
    Sn, On, cn, _, kn = coef(rn)
    s1 = Sc * rc**3 * Oc
    a_n = 4.0 / (Sn * rn**3 * cn * On)

    size = 2 * N - 1
    A = np.zeros((size, size), dtype=complex)
    j = np.arange(N)
    A[j, j] = zc
    cf = 1j / (s1 * dr)
    jj = j[:-1]
    A[jj, N + jj] = cf[jj]                     # G_{j+1}
    jj = j[1:]
    A[jj, N + jj - 1] = -cf[jj]                # G_j
    i = np.arange(1, N)
    row = N + i - 1
    cg = 1j / (a_n * (rc[i] - rc[i - 1]))
    A[row, row] = kn - 1j * alpha * On
    A[row, i] = cg                             # W_i
    A[row, i - 1] = -cg                        # W_{i-1}

    w, V = np.linalg.eig(A)
    W = V[:N]
    ws = s1 * dr
    flat = np.abs(ws @ W)**2 / (ws.sum() * (ws @ np.abs(W)**2))
    if return_all:
        return w, flat
    k = int(np.argmax(flat))
    return w[k], float(flat[k])


def _kerr_coef(a, r_in, p, q, HR_ref, rate):
    """coef(r) for warp_eigenmode in Kerr (M = 1, Sigma_0 = 1, R in R_g)."""
    cs0 = C * HR_ref * float(r_isco(a))**(q - 0.5)
    two_pi = 2.0 * np.pi

    def coef(rr):
        gphi = g_phi(rr, a)
        gth = g_theta(rr, a)
        gr_ = g_r(rr, a)
        Om = two_pi * gphi
        if rate == 'lt':
            zr = two_pi * (gphi - gth)
            kr = two_pi * (gphi - gr_)
        else:
            zr = 0.5 * Om * (1.0 - (gth / gphi)**2)
            kr = 0.5 * Om * (1.0 - (gr_ / gphi)**2)
        Sig = sigma_torque_zero(rr, r_in, p)
        cs2 = cs0**2 * rr**(-2.0 * q) / Rg_SUN**2
        return Sig, Om, cs2, zr, kr
    return coef


def _eig_point(args):
    a, r_in, r_out, p, q, HR_ref, alpha, n_eig, rate = args
    om, flat = warp_eigenmode(r_in, r_out, _kerr_coef(a, r_in, p, q, HR_ref, rate),
                              alpha, n_eig)
    return om, flat


def _kernel_eig(a, r_in, r_out, p, q, HR_ref, alpha, n_eig, rate, workers=1,
                bar=None):
    """(g, tau_w, tau_a, flatness) from the full eigenmode, per point."""
    todo = [k for k in range(a.size) if a[k] != 0.0]    # a = 0: no precession
    args = [(float(a[k]), float(r_in[k]), float(r_out[k]), p, q, HR_ref, alpha,
             n_eig, rate) for k in todo]
    res = []
    if workers and workers > 1 and len(args) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for out in ex.map(_eig_point, args,
                              chunksize=max(1, len(args) // (20 * workers))):
                res.append(out)
                if bar is not None:
                    bar.update(1)
    else:
        for t in args:
            res.append(_eig_point(t))
            if bar is not None:
                bar.update(1)
    om = np.zeros(a.size, dtype=complex)                # a = 0: omega = 0, g = 0
    flat = np.full(a.size, np.nan)
    for k, (o, f) in zip(todo, res):
        om[k], flat[k] = o, f
    g = np.abs(om.real) / (2.0 * np.pi)
    gam = -om.imag
    tau_a = np.where(gam > 0.0, 1.0 / np.where(gam > 0.0, gam, 1.0), np.nan)
    r_isco_a = r_isco(a)
    tau_w = (2.0 * Rg_SUN / C / HR_ref * r_isco_a**(0.5 - q)
             * (r_out**(1.0 + q) - r_in**(1.0 + q)) / (1.0 + q))
    return g, tau_w, tau_a, flat


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
                  HR_ref=HR_ISCO_DEFAULT, alpha=ALPHA, n_rad=N_RAD,
                  closure=CLOSURE_DEFAULT, method=METHOD_DEFAULT,
                  rate=RATE_DEFAULT, n_eig=N_EIG, workers=1,
                  diagnostics=False, progress=False):
    """
    (g, tau_w, tau_a) at M = 1, alpha = ALPHA, HR_isco = HR_ref.

    a, r_out, r_in : array_like, mutually broadcastable (r_in = r_isco(a) if
    None). Evaluation is chunked, so arbitrarily large (a, r_out) grids fit
    in memory. Points with r_out <= r_in are NaN.
    method : 'motta' (closed-form first order, uses `closure`) or 'eig' (full
    eigenmode, uses `rate`, `n_eig`, `workers`); see the module docstring.
    diagnostics : with method='motta', store the first-order validity
    diagnostics in PIFGeometric.extra; with method='eig' the flatness of the
    selected mode is always stored.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if closure not in CLOSURES:
        raise ValueError(f"closure must be one of {CLOSURES}, got {closure!r}")
    if rate not in RATES:
        raise ValueError(f"rate must be one of {RATES}, got {rate!r}")
    if method == 'motta':
        n_out = 6 if diagnostics else 3
        outs = _evaluate(
            lambda a_, ri_, ro_: _kernel_full(a_, ri_, ro_, p, q, HR_ref, alpha,
                                              n_rad, closure, diagnostics),
            n_out, a, r_out, r_in, n_rad)
        extra = {}
        if diagnostics:
            extra = dict(dl_theta=outs[3], psi_max=outs[4],
                         gamma_over_alpha_omega=outs[5])
        return PIFGeometric(outs[0], outs[1], outs[2], HR_ref, alpha,
                            'motta', extra)

    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None
        if tqdm is not None:
            a_b, ri_b, ro_b = np.broadcast_arrays(
                np.asarray(a, dtype=float),
                r_isco(a) if r_in is None else np.asarray(r_in, dtype=float),
                np.asarray(r_out, dtype=float))
            total = int(np.count_nonzero((ro_b > ri_b) & (a_b != 0.0)))
            bar = tqdm(total=total, desc="eigenmodes", unit="pt")
    try:
        outs = _evaluate(
            lambda a_, ri_, ro_: _kernel_eig(a_, ri_, ro_, p, q, HR_ref, alpha,
                                             n_eig, rate, workers, bar),
            4, a, r_out, r_in, 2000)
    finally:
        if bar is not None:
            bar.close()
    return PIFGeometric(outs[0], outs[1], outs[2], HR_ref, alpha, 'eig',
                        dict(flatness=outs[3]))


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