# projection_plots.py
#
# Proiezioni pairwise (a,R), (a,M), (R,M) della regione dei parametri
# ammessa da un match a singola frequenza nu0, marginalizzando sul terzo
# parametro.
#
# Costruzione:
#   - Proiezione (a,R), M in [M_lo,M_hi] marginalizzata:
#       esiste M ammissibile  <=>  g_X(R,a) in [M_lo*nu0, M_hi*nu0]
#     (implicazione diretta, nessuna riduzione necessaria)
#   - Proiezione (a,M), R in [r_isco(a), R_MAX] marginalizzato:
#       esiste R ammissibile <=> M*nu0 in [Gmin_a(a), Gmax_a(a)]
#     dove Gmin_a(a) = min_R g_X(R,a), Gmax_a(a) = max_R g_X(R,a)
#   - Proiezione (R,M), a in [-A_THORNE,A_THORNE] marginalizzato:
#       analoga, con Gmin_R(R) = min_a g_X(R,a), Gmax_R(R) = max_a g_X(R,a)
#
# Gmin/Gmax NON assumono monotonia (rilevante per nu_LT): sono min/max
# numerici sulla griglia. Vengono calcolati UNA sola volta per canale
# (indipendenti da nu0 e dalla sorgente); il costo per sorgente e' poi
# O(N) (sole divisioni per nu0), quindi l'aggiunta di sorgenti e'
# essenzialmente gratuita.

import warnings
import numpy as np
from setup import r_isco, A_THORNE, R_MAX, M_AGN_MIN, M_AGN_MAX


class ChannelGrid:
    """
    Precalcolo per canale: griglia g_X(R,a) mascherata (R>=r_isco(a)) e le
    quattro riduzioni Gmin_a, Gmax_a, Gmin_R, Gmax_R. Si costruisce una
    volta per canale e si riusa per tutte le sorgenti/proiezioni.
    """
    def __init__(self, g_func, a_vals, r_vals):
        self.a_vals = a_vals
        self.r_vals = r_vals
        A, R = np.meshgrid(a_vals, r_vals, indexing="ij")  # shape (Na, Nr)
        G = g_func(R, A)
        G = np.where(R >= r_isco(A), G, np.nan)
        self.A, self.R, self.G = A, R, G

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            self.Gmin_a = np.nanmin(G, axis=1)   # riduzione su R -> shape (Na,)
            self.Gmax_a = np.nanmax(G, axis=1)
            self.Gmin_R = np.nanmin(G, axis=0)   # riduzione su a -> shape (Nr,)
            self.Gmax_R = np.nanmax(G, axis=0)


##########################################################
# --------------------------------------------------------
##########################################################
def plot_projection_aR(ax, chgrid, nu0, mass_range, color, ls="-", alpha=0.15, label=None):
    """Pannello (a,R): banda/linea dove g_X(R,a) in [M_lo*nu0, M_hi*nu0]."""
    M_lo, M_hi = mass_range if mass_range is not None else (M_AGN_MIN, M_AGN_MAX)
    if M_lo == M_hi:
        ax.contour(chgrid.A, chgrid.R, chgrid.G, levels=[M_lo*nu0],
                   colors=[color], linewidths=1, linestyles=ls)
    else:
        ax.contourf(chgrid.A, chgrid.R, chgrid.G, levels=[M_lo*nu0, M_hi*nu0],
                    colors=[color], alpha=alpha)
        ax.contour(chgrid.A, chgrid.R, chgrid.G, levels=[M_lo*nu0, M_hi*nu0],
                   colors=[color], linewidths=1)
    if label:
        ax.plot([], [], color=color, lw=4, alpha=1, ls=ls, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("a"); ax.set_ylabel(r"R [$R_g$]")


def plot_projection_aM(ax, chgrid, nu0, color, ls="-", alpha=0.15, label=None):
    """Pannello (a,M): banda M in [Gmin_a(a)/nu0, Gmax_a(a)/nu0]."""
    M_lo = chgrid.Gmin_a / nu0
    M_hi = chgrid.Gmax_a / nu0
    valid = np.isfinite(M_lo) & np.isfinite(M_hi)
    ax.fill_between(chgrid.a_vals[valid], M_lo[valid], M_hi[valid],
                    color=color, alpha=alpha)
    ax.plot(chgrid.a_vals[valid], M_lo[valid], color=color, lw=1.5, ls=ls)
    ax.plot(chgrid.a_vals[valid], M_hi[valid], color=color, lw=1.5, ls=ls)
    if label:
        ax.plot([], [], color=color, lw=4, alpha=1, ls=ls, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("a"); ax.set_ylabel(r"M [$M_\odot$]")
    ax.set_xlim(-1,1)


def plot_projection_RM(ax, chgrid, nu0, color, ls="-", alpha=0.15, label=None):
    """Pannello (R,M): banda M in [Gmin_R(R)/nu0, Gmax_R(R)/nu0]."""
    M_lo = chgrid.Gmin_R / nu0
    M_hi = chgrid.Gmax_R / nu0
    valid = np.isfinite(M_lo) & np.isfinite(M_hi)
    ax.fill_between(chgrid.r_vals[valid], M_lo[valid], M_hi[valid],
                    color=color, alpha=alpha)
    ax.plot(chgrid.r_vals[valid], M_lo[valid], color=color, lw=1.5, ls=ls)
    ax.plot(chgrid.r_vals[valid], M_hi[valid], color=color, lw=1.5, ls=ls)
    if label:
        ax.plot([], [], color=color, lw=4, alpha=1, ls=ls, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"R [$R_g$]"); ax.set_ylabel(r"M [$M_\odot$]")