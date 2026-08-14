"""
modes_scan.py

Scan (a, M) per il p-mode fondamentale (m=0, n=0) a nu0 fissata:
per ciascun punto della griglia calcola se esiste una finestra di
trapping [r_isco, r1] (condizione kappa(r)=omega, Kato Fig. 6), e se
si' t_wave (Eq. 6 Motta) e t_visc_Kato (Eq. 73 Kato) valutato al CENTRO
della finestra.

Perche' al centro e non al bordo
----------------------------------
G(r) -> 0 per costruzione quando r -> r1 (perche' r1 e' definito da
kappa(r1)=omega, e G ~ (omega^2-kappa(r)^2)), quindi t_visc diverge
sempre al bordo esterno della finestra: e' un effetto di bordo del
turning point (dove peraltro l'approssimazione WKB alla base di tutta
la derivazione di Kato diventa essa stessa marginale), non un'informazione
fisica sulla tipica scala temporale del modo. Il centro della finestra
e' una scelta di rappresentativita' arbitraria ma ragionevole, lontana
da entrambi i bordi singolari (a r_isco, kappa=0 esattamente, quindi
G e' finito ma comunque un caso limite).

Design computazionale
----------------------
Il root-finding di r1 (kappa(r)=omega) richiede uno scan radiale non
vettorizzabile in forma chiusa: e' l'unica parte del calcolo fatta con
un doppio loop Python su (a, M). Tutto il resto (t_wave, t_visc al
centro finestra) e' vettorizzato dopo che la griglia di r1 e' nota.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from setup import r_isco, nu_r, set_style, fix_spines, NU0, M_AGN_MIN, M_AGN_MAX
from disk_profiles import t_wave_closed
from growth_rate import growth_rate_p_mode


def _r1_and_kappamax_scalar(a, M, nu0, r_scan_max, n_scan):
    """Root-finding scalare: r1 (bordo trapping) e kappa_max, in un solo
    scan radiale (kappa_max e' un sottoprodotto gratuito)."""
    r_in = r_isco(a)
    r_scan = np.geomspace(r_in * 1.0001, r_scan_max, n_scan)
    kappa_scan = nu_r(r_scan, a, M)
    kappa_max = kappa_scan.max()

    if nu0 >= kappa_max:
        return np.nan, kappa_max  # nessun trapping

    diff = nu0 - kappa_scan
    sign_changes = np.where(np.diff(np.sign(diff)) < 0)[0]
    if len(sign_changes) == 0:
        return np.nan, kappa_max

    i = sign_changes[0]
    denom = diff[i + 1] - diff[i]
    if denom == 0:
        r1 = r_scan[i]
    else:
        r1 = r_scan[i] - diff[i] * (r_scan[i + 1] - r_scan[i]) / denom
    return r1, kappa_max


def scan_p_mode_grid(a_vals, M_vals, nu0, alpha, r_scan_max=5000.0, n_scan=3000):
    """
    Griglia (a, M) [meshgrid indexing='ij'] per il p-mode fondamentale a
    nu0 fissata.

    Ritorna un dict con, ciascuno shape (len(a_vals), len(M_vals)):
      r_isco_grid, r1_grid, kappa_max_grid : geometria/trapping (NaN se
          non esiste trapping)
      t_wave_grid, t_visc_mid_grid, G_mid_grid : tempi scala (NaN dove
          non c'e' trapping)
      trapped_mask : bool, True dove esiste una finestra di trapping
    """
    NA, NM = len(a_vals), len(M_vals)
    r1_grid = np.full((NA, NM), np.nan)
    kappa_max_grid = np.full((NA, NM), np.nan)

    # --- unica parte non vettorizzabile: root-finding di r1 ---
    for i, a in enumerate(a_vals):
        for j, M in enumerate(M_vals):
            r1, kmax = _r1_and_kappamax_scalar(a, M, nu0, r_scan_max, n_scan)
            r1_grid[i, j] = r1
            kappa_max_grid[i, j] = kmax

    trapped_mask = np.isfinite(r1_grid)

    # --- resto vettorizzato sull'intera griglia ---
    A_grid, M_grid = np.meshgrid(a_vals, M_vals, indexing='ij')
    r_isco_grid = r_isco(A_grid)
    r_mid_grid = 0.5 * (r_isco_grid + r1_grid)  # NaN dove non trapped, propaga correttamente

    t_wave_grid = t_wave_closed(r_isco_grid, r1_grid, A_grid, M_grid)

    G_mid_grid = growth_rate_p_mode(r_mid_grid, A_grid, M_grid, nu0, alpha)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_visc_mid_grid = 1.0 / np.abs(G_mid_grid)

    return dict(
        r_isco_grid=r_isco_grid, r1_grid=r1_grid, kappa_max_grid=kappa_max_grid,
        t_wave_grid=t_wave_grid, t_visc_mid_grid=t_visc_mid_grid,
        G_mid_grid=G_mid_grid, trapped_mask=trapped_mask,
    )



"""
Mappa (a, M) per il p-mode fondamentale a nu0 fissata (J1257): dove
esiste una finestra di trapping, e come si comportano t_wave e
t_visc_Kato (valutato al centro della finestra) rispetto a P_osc=1/nu0.

Griglia 100x100, root-finding di r1 non vettorizzato (~1.7s), resto
vettorizzato.
"""

if __name__ == "__main__":
    set_style()

    ALPHA = 0.1
    NU_TARGET = NU0
    N_A, N_M = 100, 100

    a_vals = np.linspace(-0.998, 0.998, N_A)
    M_vals = np.logspace(5, 10, N_M)

    res = scan_p_mode_grid(a_vals, M_vals, NU_TARGET, ALPHA, n_scan=4000)

    A_grid, M_grid = np.meshgrid(a_vals, M_vals, indexing='ij')
    Posc = 1.0 / NU_TARGET
    trapped = res['trapped_mask']

    ratio_wave = np.where(trapped, res['t_wave_grid'] / Posc, np.nan)
    ratio_visc = np.where(trapped, res['t_visc_mid_grid'] / Posc, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax in axes:
        fix_spines(ax)
        ax.set_yscale('log')
        ax.set_xlabel(r"$a$")
        ax.set_xlim(-1, 1)

    # --- Pannello 1: esistenza di trapping + larghezza relativa finestra ---
    ax = axes[0]
    width_rel = np.where(trapped, (res['r1_grid'] - res['r_isco_grid']) / res['r_isco_grid'], np.nan)
    pcm = ax.pcolormesh(A_grid, M_grid, np.log10(width_rel), shading='auto', cmap='viridis')
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\log_{10}[(r_1-r_{\rm isco})/r_{\rm isco}]$")
    ax.set_title("Larghezza relativa finestra di trapping")
    ax.set_ylabel(r"$M\ [M_\odot]$")

    # --- Pannello 2: t_wave / P_osc ---
    ax = axes[1]
    pcm = ax.pcolormesh(A_grid, M_grid, np.log10(ratio_wave), shading='auto', cmap='RdBu_r',
                        vmin=-3, vmax=3)
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\log_{10}(t_{\rm wave}/P_{\rm osc})$")
    ax.contour(A_grid, M_grid, ratio_wave, levels=[1.0], colors='black', linewidths=1.2)
    ax.set_title(r"$t_{\rm wave}/P_{\rm osc}$ (contorno: $=1$)")

    # --- Pannello 3: t_visc(centro finestra) / P_osc ---
    ax = axes[2]
    pcm = ax.pcolormesh(A_grid, M_grid, np.log10(ratio_visc), shading='auto', cmap='RdBu_r',
                        vmin=-3, vmax=3)
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\log_{10}(t_{\rm visc}^{\rm Kato}/P_{\rm osc})$")
    ax.contour(A_grid, M_grid, ratio_visc, levels=[1.0], colors='black', linewidths=1.2)
    ax.set_title(r"$t_{\rm visc}^{\rm Kato}$(centro)$/P_{\rm osc}$ (contorno: $=1$)")

    fig.suptitle(fr"p-mode fondamentale, $\nu_0={NU_TARGET:.2e}$ Hz (J1257), $\alpha={ALPHA}$",
                fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('output_modes/modes_scan_aM_J1257.pdf', bbox_inches='tight')
    print("Salvato: modes_scan_aM_J1257.pdf")

    # --- riepilogo numerico ---
    n_tot = trapped.size
    n_trap = trapped.sum()
    n_wave_ok = np.nansum(ratio_wave < 1.0)
    n_visc_ok = np.nansum(ratio_visc > 1.0)
    n_both = np.nansum((ratio_wave < 1.0) & (ratio_visc > 1.0))
    print(f"\nPunti griglia totali: {n_tot}")
    print(f"Punti con trapping: {n_trap} ({100*n_trap/n_tot:.1f}%)")
    print(f"  di cui t_wave<P_osc: {n_wave_ok} ({100*n_wave_ok/n_trap:.1f}% dei trapped)")
    print(f"  di cui t_visc>P_osc: {n_visc_ok} ({100*n_visc_ok/n_trap:.1f}% dei trapped)")
    print(f"  di cui ENTRAMBE:     {n_both} ({100*n_both/n_trap:.1f}% dei trapped)")
    print(f"G negativo (smorzamento) da qualche parte? {np.any(res['G_mid_grid'][trapped] < 0)}")