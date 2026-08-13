"""
modes_timescales.py

Analogo di Fig. 7 di Motta et al. 2017 per il p-mode fondamentale
(m=0, n=0) di Kato: t_visc_Kato(r) (curva, Eq. 73 semplificata) vs
t_wave (retta, integrale sull'intera finestra di trapping [r_isco,r1])
vs P_osc=1/nu0 (retta), a spin e massa fissati.

Il bordo r1 della finestra di trapping e' determinato dalla condizione
kappa(r)=omega (Kato, Fig. 6, Sez. 5.1): se nu0 > kappa_max, non esiste
trapping per il p-mode fondamentale a questi (a, M) e lo script lo
segnala esplicitamente invece di produrre un grafico vuoto/fuorviante.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from setup import r_isco, nu_r, set_style, fix_spines, NU0
from disk_profiles import t_wave_closed
from growth_rate import growth_rate_p_mode, find_p_mode_outer_boundary

set_style()

# --- parametri fissati per questo confronto ---
A_SPIN = 0.5
M_BH_TEST = 2e6        # M_sun, ordine di grandezza compatibile con J1257
ALPHA = 0.1            # viscosita' turbolenta (Kato usa 0.1-0.3 in Fig. 12)
NU_TARGET = NU0        # 3.3e-5 Hz, J1257

r_in = r_isco(A_SPIN)
r_scan_max = 5000.0  # AGN: raggi molto piu' estesi che per XRB
kappa_max = nu_r(np.geomspace(r_in * 1.001, r_scan_max, 20000), A_SPIN, M_BH_TEST).max()

print(f"kappa_max(a={A_SPIN}, M={M_BH_TEST:.1e} Msun) = {kappa_max:.3e} Hz")
print(f"nu0 (target) = {NU_TARGET:.3e} Hz")

if NU_TARGET >= kappa_max:
    print("ATTENZIONE: nu0 >= kappa_max -- nessuna finestra di trapping del "
          "p-mode fondamentale esiste per questi (a, M). "
          "Il p-mode non puo' spiegare un QPO a questa frequenza con questa "
          "massa/spin: aumentare M (kappa_max ~ 1/M) o cambiare a.")
else:
    r1 = find_p_mode_outer_boundary(A_SPIN, M_BH_TEST, NU_TARGET,
                                     r_scan_max=r_scan_max, n_scan=20000)
    print(f"Finestra di trapping p-mode: [r_isco={r_in:.3f}, r1={r1:.3f}] Rg")

    r_grid = np.linspace(r_in * 1.001, r1 * 0.999, 300)
    G = growth_rate_p_mode(r_grid, A_SPIN, M_BH_TEST, NU_TARGET, ALPHA)
    t_visc = 1.0 / np.abs(G)

    t_wave = t_wave_closed(r_in, r1, A_SPIN, M_BH_TEST)
    P_osc = 1.0 / NU_TARGET

    print(f"t_wave [s] = {t_wave:.3e}")
    print(f"P_osc [s]  = {P_osc:.3e}")
    print(f"t_visc range [s] = [{np.nanmin(t_visc):.3e}, {np.nanmax(t_visc):.3e}]")

    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    fix_spines(ax)

    ax.plot(r_grid, t_visc, color='#C0392B', lw=1.5,
            label=r"$t_{\rm visc}^{\rm Kato}(r)$")
    ax.axhline(t_wave, color='steelblue', ls='--', lw=1.5,
               label=r"$t_{\rm wave}$ (finestra intera)")
    ax.axhline(P_osc, color='black', ls=':', lw=1.5,
               label=r"$P_{\rm osc}=1/\nu_0$")

    ax.axvline(r_in, color='purple', ls='--', lw=0.8, alpha=0.6)
    ax.axvline(r1, color='gray', ls='--', lw=0.8, alpha=0.6)

    ax.set_xlabel(r"$r$ [$R_g$]")
    ax.set_ylabel("Tempo [s]")
    ax.set_yscale('log')
    ax.set_title(fr"p-mode fondamentale, $a={A_SPIN}$, $M={M_BH_TEST:.1e}\,M_\odot$ — J1257")
    ax.legend(loc='best', frameon=True, fontsize=7.5)

    plt.tight_layout()
    plt.savefig('/home/claude/tscales/modes_timescales_J1257.pdf', bbox_inches='tight')
    print("Salvato: modes_timescales_J1257.pdf")
    plt.close()
