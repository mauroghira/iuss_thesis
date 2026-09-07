# projection_plots.py
#
# Proiezioni pairwise (a,R), (a,M), (R,M) della regione dei parametri
# ammessa da un match a singola frequenza nu0, marginalizzando sul terzo
# parametro.
#
# Costruzione generale (invariata rispetto alla versione precedente):
#   - Proiezione (a,R): esiste M ammissibile in [M_lo,M_hi]
#         <=> g_X(R,a) in [M_lo*nu0, M_hi*nu0]
#     (implicazione diretta: la mappa M(R,a) = g_X(R,a)/nu0 e' univoca a
#     nu0 fissata, quindi il vincolo su M si traduce esattamente in un
#     vincolo di livello su g_X, senza bisogno di riduzioni.)
#   - Proiezione (a,M): esiste R ammissibile in [r_isco(a), R_MAX]
#         <=> M*nu0 in [Gmin_a(a), Gmax_a(a)]
#     dove Gmin_a(a) = min_R g_X(R,a), Gmax_a(a) = max_R g_X(R,a)
#     (nessuna monotonia assunta: min/max numerici sulla griglia, rilevante
#     in particolare per nu_LT che puo' avere un massimo locale non
#     all'ISCO, si veda setup.py e param_space.extremal_over_axis).
#   - Proiezione (R,M): analoga, con Gmin_R(R) = min_a g_X(R,a),
#     Gmax_R(R) = max_a g_X(R,a).
#
# Gmin/Gmax vengono calcolati UNA sola volta per canale in ChannelGrid
# (indipendenti da nu0 e dalla sorgente, dipendono solo dalla geometria
# g_X e dai vincoli ISCO/Thorne gia' imposti sulla griglia (a,R)); il
# costo per sorgente e' poi O(N) (sole divisioni per nu0), quindi
# l'aggiunta di sorgenti e' essenzialmente gratuita.
#
# --------------------------------------------------------------------
# NUOVA DISTINZIONE (Regione A vs Regione B)
# --------------------------------------------------------------------
# Per ciascuna sorgente si distinguono ora ESPLICITAMENTE due regioni
# ammesse dalla stessa equazione g_X(R,a) = M*nu0, che differiscono solo
# per il prior su M usato:
#
#   Regione A ("accessibile", linea continua)
#       M in [M_AGN_MIN, M_AGN_MAX]  (limite di Thorne 1974 per
#       l'accrescimento, generico per qualunque AGN — NON e' informazione
#       specifica sulla sorgente, e' un vincolo fisico universale del
#       modello). Usata SEMPRE, indipendentemente dal fatto che la
#       sorgente abbia o meno una stima di massa indipendente: rappresenta
#       "cosa e' raggiungibile riproducendo nu0, senza usare alcuna
#       informazione esterna (di letteratura) sulla singola sorgente".
#
#   Regione B ("vincolata dalla massa nota", linea tratteggiata)
#       M in mass_range, la stima di massa indipendente riportata nel
#       testo della tesi per QUELLA sorgente (catalog.py). Esiste solo se
#       mass_range non e' None. Se mass_range e' una stima puntuale
#       (lo==hi, convenzione di catalog.py), B degenera in una curva di
#       livello (area nulla nel piano): si disegna la linea tratteggiata
#       ma non si tenta alcun riempimento, perche' l'intersezione di
#       un'area con un insieme di misura nulla ha essa stessa misura
#       nulla — non e' un'approssimazione, e' l'esatto risultato
#       geometrico.
#
# Il riempimento a colore rappresenta SEMPRE e SOLO l'intersezione
# Regione A ∩ Regione B. Se mass_range e' None, non esiste un vincolo B
# aggiuntivo rispetto ad A: si riempie direttamente Regione A (e' la
# migliore regione disponibile in assenza di informazione di letteratura
# sulla sorgente). Non si assume mai B ⊆ A: l'intersezione viene sempre
# calcolata esplicitamente, cosi' il codice resta corretto anche nel
# caso (non atteso, ma non verificato a priori) in cui una stima di
# letteratura ecceda il range AGN generico.

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

def _area_from_band(x, y_lo, y_hi, valid, log_x=False, log_y=False):
    """Area in the coordinates used by the corresponding plotted axes."""
    x = np.asarray(x)
    y_lo = np.asarray(y_lo)
    y_hi = np.asarray(y_hi)
    valid = np.asarray(valid, dtype=bool)

    if not np.any(valid):
        return 0.0

    xp = np.log(x) if log_x else x
    lop = np.log(y_lo) if log_y else y_lo
    hip = np.log(y_hi) if log_y else y_hi

    good = valid & np.isfinite(xp) & np.isfinite(lop) & np.isfinite(hip)
    if np.count_nonzero(good) < 2:
        return 0.0

    return float(np.trapezoid(hip[good] - lop[good], xp[good]))


def _area_from_mask_2d(A, R, mask, log_x=False, log_y=False):
    """
    Approximate the area of a 2-D Boolean region in plotted coordinates.
    Used for (a,R), where contourf operates directly on the mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return 0.0

    xp = np.log(A[:, 0]) if log_x else A[:, 0]
    yp = np.log(R[0, :]) if log_y else R[0, :]

    dx = np.abs(np.diff(xp))
    dy = np.abs(np.diff(yp))

    cells = (
        mask[:-1, :-1]
        & mask[1:, :-1]
        & mask[:-1, 1:]
        & mask[1:, 1:]
    )

    return float(np.sum(cells * dx[:, None] * dy[None, :]))


def _assign_alpha_by_area(areas, alpha_min=0.2, alpha_max=0.5):
    """Return indices sorted by decreasing area and their alphas."""
    areas = np.asarray(areas, dtype=float)
    order = np.argsort(-areas, kind="stable")

    if len(order) == 1:
        return order, np.array([alpha_max], dtype=float)

    return order, np.linspace(alpha_min, alpha_max, len(order))


def plot_regions_by_area(regions, alpha_min=0.2, alpha_max=0.5):
    """
    Draw region descriptors in decreasing-area order.

    A region descriptor is the dictionary returned by one of the
    plot_projection_* functions with return_region=True.
    """
    if not regions:
        return []

    areas = np.array([r["area"] for r in regions], dtype=float)
    order, alphas = _assign_alpha_by_area(
        areas, alpha_min=alpha_min, alpha_max=alpha_max
    )

    result = []
    for idx, alpha in zip(order, alphas):
        regions[idx]["draw"](float(alpha))
        result.append((idx, float(areas[idx]), float(alpha)))

    return result


def plot_projection_aR(ax, chgrid, nu0, mass_range, color, alpha=0.35, label=None, draw_fill=True, return_region=False):
    """
    Pannello (a,R).

    Regione A (continua): g_X(R,a) in [M_AGN_MIN*nu0, M_AGN_MAX*nu0].
    Regione B (tratteggiata, se mass_range dato): g_X(R,a) in
        [M_lo*nu0, M_hi*nu0].
    Riempimento: maschera booleana A & B (o solo A se mass_range=None),
    disegnata con contourf su un campo 0/1 -- non si assume alcuna
    relazione di inclusione tra A e B, l'intersezione e' calcolata
    esplicitamente punto per punto sulla griglia.
    """
    G, A, R = chgrid.G, chgrid.A, chgrid.R

    levels_A = [M_AGN_MIN * nu0, M_AGN_MAX * nu0]
    ax.contour(A, R, G, levels=levels_A, colors=[color], linewidths=1, linestyles='-')
    mask_A = (G >= levels_A[0]) & (G <= levels_A[1])  # NaN -> False automaticamente

    mask_fill = mask_A
    if mass_range is not None:
        M_lo, M_hi = mass_range
        if M_lo == M_hi:
            # stima puntuale: solo curva di livello, nessun riempimento
            # (area nulla, si veda nota in testa al file)
            ax.contour(A, R, G, levels=[M_lo * nu0], colors=[color],
                       linewidths=1, linestyles=':')
            mask_fill = np.zeros_like(mask_A)
        else:
            levels_B = [M_lo * nu0, M_hi * nu0]
            ax.contour(A, R, G, levels=levels_B, colors=[color], linewidths=1, linestyles=':')
            mask_B = (G >= levels_B[0]) & (G <= levels_B[1])
            mask_fill = mask_A & mask_B

    area = _area_from_mask_2d(
        A, R, mask_fill, log_x=False, log_y=True
    ) if mask_fill.any() else 0.0

    def draw_fill(fill_alpha):
        if mask_fill.any():
            ax.contourf(
                A, R, mask_fill.astype(float),
                levels=[0.5, 1.5],
                colors=[color],
                alpha=fill_alpha,
            )

    if draw_fill:
        draw_fill(alpha)

    if label:
        ax.plot([], [], color=color, lw=4, alpha=1, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("a"); ax.set_ylabel(r"R [$R_g$]")

    if return_region:
        return {"area": area, "draw": draw_fill, "label": label, "color": color}
    return None


##########################################################
# --------------------------------------------------------
##########################################################
def plot_projection_aM(ax, chgrid, nu0, mass_range, color, alpha=0.5, label=None, draw_fill=True, return_region=False):
    """
    Pannello (a,M).

    Banda "grezza" raggiungibile variando R (unico vincolo: fisico,
    R in [r_isco(a), R_MAX], gia' incorporato in Gmin_a/Gmax_a):
        [Gmin_a(a)/nu0, Gmax_a(a)/nu0]

    Regione A (continua) = banda grezza ∩ [M_AGN_MIN, M_AGN_MAX].
    Regione B (tratteggiata, se mass_range dato) = banda grezza ∩ mass_range.
    Riempimento = A ∩ B (intersezione di intervalli, esatta -- non e'
    necessario assumere B subset A).
    """
    M_lo_raw = chgrid.Gmin_a / nu0
    M_hi_raw = chgrid.Gmax_a / nu0
    valid_raw = np.isfinite(M_lo_raw) & np.isfinite(M_hi_raw)

    A_lo = np.maximum(M_lo_raw, M_AGN_MIN)
    A_hi = np.minimum(M_hi_raw, M_AGN_MAX)
    valid_A = valid_raw & (A_lo <= A_hi)

    ax.plot(chgrid.a_vals[valid_A], A_lo[valid_A], color=color, lw=1.2, ls='-')
    ax.plot(chgrid.a_vals[valid_A], A_hi[valid_A], color=color, lw=1.2, ls='-')

    fill_lo, fill_hi, valid_fill = None, None, None

    if mass_range is not None:
        M_lo, M_hi = mass_range
        if M_lo == M_hi:
            # stima puntuale: retta orizzontale tratteggiata sull'intero
            # pannello (il valore noto non dipende da a); nessun
            # riempimento (area nulla, si veda nota in testa al file).
            ax.axhline(M_lo, color=color, lw=1, ls=':')
        else:
            # rette orizzontali A TUTTA LARGHEZZA: il bound noto e' una
            # costante indipendente da a, quindi non va clippato alla
            # banda grezza raggiungibile -- il clipping serve SOLO al
            # calcolo del riempimento (dove il punto e' fisicamente
            # realizzabile), non alla rappresentazione della soglia B.
            ax.axhline(M_lo, color=color, lw=1, ls=':')
            ax.axhline(M_hi, color=color, lw=1, ls=':')

            B_lo = np.maximum(M_lo_raw, M_lo)
            B_hi = np.minimum(M_hi_raw, M_hi)
            valid_B = valid_raw & (B_lo <= B_hi)

            fill_lo = np.maximum(A_lo, B_lo)
            fill_hi = np.minimum(A_hi, B_hi)
            valid_fill = valid_A & valid_B & (fill_lo <= fill_hi)
    else:
        fill_lo, fill_hi, valid_fill = A_lo, A_hi, valid_A

    area = _area_from_band(
        chgrid.a_vals, fill_lo, fill_hi, valid_fill,
        log_x=False, log_y=True
    ) if fill_lo is not None else 0.0

    def draw_fill(fill_alpha):
        if fill_lo is not None and valid_fill.any():
            ax.fill_between(
                chgrid.a_vals[valid_fill],
                fill_lo[valid_fill],
                fill_hi[valid_fill],
                color=color,
                alpha=fill_alpha,
            )

    if draw_fill:
        draw_fill(alpha)

    if label:
        ax.plot([], [], color=color, lw=4, alpha=1, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("a"); ax.set_ylabel(r"M [$M_\odot$]")
    ax.set_xlim(-1, 1)

    if return_region:
        return {"area": area, "draw": draw_fill, "label": label, "color": color}
    return None


##########################################################
# --------------------------------------------------------
##########################################################
def plot_projection_RM(ax, chgrid, nu0, mass_range, color, alpha=0.5, label=None, draw_fill=True, return_region=False):
    """
    Pannello (R,M). Stessa logica di plot_projection_aM, con la
    riduzione su a (Gmin_R, Gmax_R) al posto della riduzione su R
    (vincolo fisico gia' incorporato: |a|<=A_THORNE).
    """
    M_lo_raw = chgrid.Gmin_R / nu0
    M_hi_raw = chgrid.Gmax_R / nu0
    valid_raw = np.isfinite(M_lo_raw) & np.isfinite(M_hi_raw)

    A_lo = np.maximum(M_lo_raw, M_AGN_MIN)
    A_hi = np.minimum(M_hi_raw, M_AGN_MAX)
    valid_A = valid_raw & (A_lo <= A_hi)

    ax.plot(chgrid.r_vals[valid_A], A_lo[valid_A], color=color, lw=1.2, ls='-')
    ax.plot(chgrid.r_vals[valid_A], A_hi[valid_A], color=color, lw=1.2, ls='-')

    fill_lo, fill_hi, valid_fill = None, None, None

    if mass_range is not None:
        M_lo, M_hi = mass_range
        if M_lo == M_hi:
            ax.axhline(M_lo, color=color, lw=1, ls=':')
        else:
            ax.axhline(M_lo, color=color, lw=1, ls=':')
            ax.axhline(M_hi, color=color, lw=1, ls=':')

            B_lo = np.maximum(M_lo_raw, M_lo)
            B_hi = np.minimum(M_hi_raw, M_hi)
            valid_B = valid_raw & (B_lo <= B_hi)

            fill_lo = np.maximum(A_lo, B_lo)
            fill_hi = np.minimum(A_hi, B_hi)
            valid_fill = valid_A & valid_B & (fill_lo <= fill_hi)
    else:
        fill_lo, fill_hi, valid_fill = A_lo, A_hi, valid_A

    area = _area_from_band(
        chgrid.r_vals, fill_lo, fill_hi, valid_fill,
        log_x=True, log_y=True
    ) if fill_lo is not None else 0.0

    def draw_fill(fill_alpha):
        if fill_lo is not None and valid_fill.any():
            ax.fill_between(
                chgrid.r_vals[valid_fill],
                fill_lo[valid_fill],
                fill_hi[valid_fill],
                color=color,
                alpha=fill_alpha,
            )

    if draw_fill:
        draw_fill(alpha)

    if label:
        ax.plot([], [], color=color, lw=4, alpha=1, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"R [$R_g$]"); ax.set_ylabel(r"M [$M_\odot$]")

    if return_region:
        return {"area": area, "draw": draw_fill, "label": label, "color": color}
    return None