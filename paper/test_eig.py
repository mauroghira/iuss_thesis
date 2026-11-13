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
    for bad in (dict(method='x'), dict(closure='x'), dict(rate='x'),
                dict(vgrav='x')):
        try:
            pc.pif_geometric(0.5, 20.0, **bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_sound_speed():
    """vgrav='kepler': numerical int 2 dR / c_s = Motta Eq. 6; vgrav='kerr':
    same t_wave in both methods, -> kepler for a -> 0, inf for a = 1."""
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    ro = np.array([3.0, 30.0, 200.0])
    ri = np.full(3, float(pc.r_isco(0.9)))
    R = pc._r_grid(ri, ro, pc.N_RAD)
    num = trapz(2 * pc.Rg_SUN / pc.sound_speed(R, 0.9, vgrav='kepler'), R, axis=-1)
    assert np.allclose(num, pc._tau_wave_closed(0.9, ri, ro, pc.Q_SOUND,
                                                pc.HR_ISCO_DEFAULT), rtol=1e-5)
    ro0 = float(pc.r_isco(1e-6)) * np.array([1.5, 5.0, 30.0])
    K = pc.pif_geometric(1e-6, ro0)
    P = pc.pif_geometric(1e-6, ro0, vgrav='kepler')
    assert np.allclose(K.tau_w, P.tau_w, rtol=1e-4)
    E = pc.pif_geometric(0.5, 20.0, method='eig', n_eig=60)
    F = pc.pif_geometric(0.5, 20.0)
    assert E.vgrav == F.vgrav == 'kerr' and np.isclose(E.tau_w, F.tau_w)
    G = pc.pif_geometric(1.0, ro)
    assert np.all(np.isinf(G.tau_w)) and np.all(np.isfinite(G.tau_a))


import faulthandler, time
#faulthandler.dump_traceback_later(40, exit=True)   # prints where it is stuck after 40 s

from pif_core import (warp_eigenmode, _kerr_coef, P_TORQUE, Q_SOUND,
                      HR_ISCO_DEFAULT, ALPHA, pif_geometric)
from pif_timescales import _loglog_at
from setup import r_isco

if __name__ == '__main__':
    """
    for f in (test_fl14_circumbinary, test_first_order_limits, test_api_guards,
              test_sound_speed):
        f()
        print('ok  ', f.__name__)
    print('all tests passed')


    r = np.geomspace(3.0, 100.0, 400)          # 400 points
    for w in (1, 2, 4, 8):
        t = time.perf_counter()
        pif_geometric(0.5, r, method='eig', n_eig=60, workers=w, progress=True)
        print('workers', w, round(time.perf_counter() - t, 1), 's', flush=True)

    

    d = np.load('output_test/pif_timescales_analysis.npz', allow_pickle=True)
    print(d['method'], d['closure'], d['n_eig'], d['a_vals'])
    for an in d['analyses']:
        T = 1.0 / an['source']['nu0']
        for m in an['panel1_matches']:
            r, tw, ta = m['r_out'], m['t_wave'], m['t_align']
            for rb, ok in zip(m['r_precession'], m['precession_ok']):
                if not ok:
                    continue
                i = np.searchsorted(r, rb)
                sl = slice(max(i - 2, 0), i + 2)
                print(f"{an['source']['name']}  a={m['parameter']:.4f}  rbar={rb:.3f}  "
                    f"1/nu0={T:.3e}  tw={_loglog_at(r, tw, rb):.3e}  "
                    f"ta={_loglog_at(r, ta, rb):.3e}")
                print("    r :", r[sl]); print("    ta:", ta[sl])


    spins = np.array([0.5, 0.9, 0.998])
    r_out = np.geomspace(5.0, 100.0, 40)       # tutti > r_isco per questi spin
    a, ri = spins[:, None], r_isco(spins)[:, None]

    # motta + lt_exact: controparte al primo ordine dell'eig con rate='lt'
    mo = pif_geometric(a, r_out[None, :], r_in=ri, method='motta',
                    closure='lt_exact', diagnostics=True)
    # eig con gli stessi alpha, HR_ref (default di pif_core)
    ei = pif_geometric(a, r_out[None, :], r_in=ri, method='eig',
                    rate='lt', n_eig=120, workers=1)

    for i, s in enumerate(spins):
        print(f"\n=== a = {s} ===")
        print(" r_out  dl_theta  psi_max  gam/(aOm)  flat  ta_eig/ta_motta  g_eig/g_motta")
        for j, r in enumerate(r_out):
            print(f"{r:7.2f} {mo.extra['dl_theta'][i,j]:9.2e} "
                f"{mo.extra['psi_max'][i,j]:8.2e} "
                f"{mo.extra['gamma_over_alpha_omega'][i,j]:9.2e} "
                f"{ei.extra['flatness'][i,j]:6.3f} "
                f"{ei.tau_a[i,j]/mo.tau_a[i,j]:12.3e} "
                f"{ei.g[i,j]/mo.g[i,j]:12.3f}")

    
    a = 0.5
    ri = float(r_isco(a))
    coef = _kerr_coef(a, ri, P_TORQUE, Q_SOUND, HR_ISCO_DEFAULT, 'lt')
    for ro in (9.98, 10.78):
        w, f = warp_eigenmode(ri, ro, coef, ALPHA, 120, return_all=True)
        k = np.argsort(-f)[:4]
        print(f"r_out={ro}")
        for i in k:
            print(f"  flat={f[i]:.3f}  g={abs(w[i].real)/(2*np.pi):.4e}  gamma={-w[i].imag:.3e}")
        
    """


    spins = np.array([0.5, 0.9, 0.998])
    a = spins[:, None]
    ri = r_isco(spins)[:, None]
    x = np.array([1.1, 1.3, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0])
    r_out = ri * x[None, :]                       # shape (3, 9), r_out > r_in

    # scansione in HR a alpha=0.01, poi in alpha a HR=0.1
    combos = [(0.03, 0.01), (0.1, 0.01), (0.3, 0.01), (0.1, 0.001), (0.1, 0.1)]

    for HR, al in combos:
        mo = pif_geometric(a, r_out, r_in=ri, HR_ref=HR, alpha=al,
                        method='motta', closure='lt_exact', diagnostics=True)
        ei = pif_geometric(a, r_out, r_in=ri, HR_ref=HR, alpha=al,
                        method='eig', rate='lt', n_eig=120)
        print(f"\n##### HR={HR}  alpha={al}")
        for i, s in enumerate(spins):
            print(f"--- a={s}  r_in={ri[i,0]:.3f}")
            print("   x   r_out  dl_theta  gam/aOm  Hr/alpha  flat  ta_eig/ta_mo  g_eig/g_mo")
            for j in range(x.size):
                print(f"{x[j]:5.2f} {r_out[i,j]:6.2f} "
                    f"{mo.extra['dl_theta'][i,j]:9.2e} "
                    f"{mo.extra['gamma_over_alpha_omega'][i,j]:8.2e} "
                    f"{HR/(x[j]*al):8.2e} "
                    f"{ei.extra['flatness'][i,j]:6.3f} "
                    f"{ei.tau_a[i,j]/mo.tau_a[i,j]:11.3e} "
                    f"{ei.g[i,j]/mo.g[i,j]:10.3f}")
            