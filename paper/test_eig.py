"""
Regression / validation tests of the eigenmode method (run: python test_pif_eig.py).

1. FL14 circumbinary benchmark (Sect. 2.4, Fig. 5): eigenmode / first-order
   ratios must fall in the band shown by FL14 (read off their Fig. 5, +-0.05).
2. First-order limits: with K -> 0, or with a large aspect ratio, the Kerr
   eigenmode must reproduce the consistent first-order solution.
3. API guards: eigen curves are not rescalable in alpha / HR_isco.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"          # forced, not setdefault
import numpy as np

import pif_core as pc


def _fl14_coef(x):
    Zin = -3 * 0.25 / 4 * 0.5**2                 # eta = 1/4, r_in = 2a  ->  -0.046875
    Om = x**-1.5
    zr = Zin * x**-2.0 * Om
    return x**-1.0, Om, 0.1**2 * x**-1.0, zr, -zr  # Sigma, Omega, cs2, ZOmega, KOmega


def _fl14_first_order(xo, alpha=0.01, n=200000):
    from scipy.integrate import cumulative_trapezoid as ct
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    x = np.geomspace(1, xo, n)
    S, Om, cs2, zr, _ = _fl14_coef(x)
    w = S * x**3 * Om
    wp = trapz(w * zr, x) / trapz(w, x)
    G = ct(w * (wp - zr), x, initial=0)
    gam = trapz(4 * alpha * G**2 / (S * cs2 * x**3), x) / trapz(w, x)
    return wp, gam


def test_fl14_circumbinary():
    for xo, (wp_lo, wp_hi), (g_lo, g_hi) in ((10, (0.85, 0.95), (0.72, 0.85)),
                                              (100, (0.78, 0.90), (0.62, 0.78)),
                                              (1000, (0.70, 0.82), (0.58, 0.70))):
        wp1, g1 = _fl14_first_order(xo)
        om, flat = pc.warp_eigenmode(1.0, float(xo), _fl14_coef, 0.01, 120)
        assert wp_lo < om.real / wp1 < wp_hi, (xo, om.real / wp1)
        assert g_lo < -om.imag / g1 < g_hi, (xo, -om.imag / g1)
        assert flat > 0.99
        om2, _ = pc.warp_eigenmode(1.0, float(xo), _fl14_coef, 0.01, 240)
        assert abs(om2.real / om.real - 1) < 1e-3     # converged in N
        assert abs(om2.imag / om.imag - 1) < 5e-3


def test_first_order_limits():
    for a, ro in ((0.1, 50.0), (0.5, 15.0), (-0.5, 50.0)):
        ri = float(pc.r_isco(a))
        # (i) K -> 0  (weak-field a: <10% off the first-order solution)
        base = pc._kerr_coef(a, ri, pc.P_TORQUE, pc.Q_SOUND, 0.1, 'lt')
        noK = lambda r: base(r)[:4] + (0.0 * base(r)[4],)
        om, _ = pc.warp_eigenmode(ri, ro, noK, pc.ALPHA, 160)
        ref = pc.pif_geometric(a, ro, closure='lt_exact')
        if a == 0.1:
            assert abs(abs(om.real) / (2 * np.pi) / float(ref.g) - 1) < 0.1
        # (ii) large aspect ratio: eigenmode -> first-order, whatever the field
        E = pc.pif_geometric(a, ro, HR_ref=3.0, method='eig')
        F = pc.pif_geometric(a, ro, HR_ref=3.0, closure='lt_exact')
        assert abs(float(E.g / F.g) - 1) < 0.02, (a, ro, float(E.g / F.g))
        assert abs(float(E.tau_a / F.tau_a) - 1) < 0.02, (a, ro)
        assert float(E.extra['flatness']) > 0.99


def test_api_guards():
    E = pc.pif_geometric(0.5, 20.0, method='eig')
    assert E.method == 'eig' and np.isfinite(E.physical(1e6)[2])
    try:
        E.physical(1e6, alpha=0.02)
    except ValueError:
        pass
    else:
        raise AssertionError('eig curves must refuse rescaling in alpha')
    for bad in (dict(method='x'), dict(closure='x'), dict(rate='x')):
        try:
            pc.pif_geometric(0.5, 20.0, **bad)
        except ValueError:
            continue
        raise AssertionError(bad)


import faulthandler, time
#faulthandler.dump_traceback_later(40, exit=True)   # prints where it is stuck after 40 s

from pif_core import pif_geometric

if __name__ == '__main__':
    """
    for f in (test_fl14_circumbinary, test_first_order_limits, test_api_guards):
        f()
        print('ok  ', f.__name__)
    print('all tests passed')
    """

    r = np.geomspace(3.0, 100.0, 400)          # 400 points
    for w in (1, 2, 4, 8):
        t = time.perf_counter()
        pif_geometric(0.5, r, method='eig', n_eig=60, workers=w, progress=True)
        print('workers', w, round(time.perf_counter() - t, 1), 's', flush=True)