"""
growth_rate.py

Tassi di crescita/smorzamento dei modi diskoseismici.

Parte 1 (p-mode, m=n=0): tasso di crescita/smorzamento viscoso G(r) di
Kato (2001), Eq. (73), generalizzato alla metrica di Kerr.

Parte 2 (c-mode, m>=1, n>=1): tasso di smorzamento corotazionale
omega_i di Tsang & Lai (2008, arXiv:0810.1299), Eq. (39) -- i c-mode
sono SEMPRE smorzati (omega_i<0, mai overstable), per un meccanismo
fisico completamente diverso dal p-mode: assorbimento dell'onda alla
risonanza di corotazione, non processo viscoso locale.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from setup import nu_r, nu_phi, nu_theta, r_isco, r_ilr, r_vertical_resonance
from disk_profiles import c_s_powerlaw


def dlnOmega_dlnr(r, a):
    """
    d ln(Omega_phi)/d ln(r), analitica, indipendente da M.
    g_phi(r,a) = cost/(r^1.5+a) => d ln g_phi/d ln r = -1.5*r^1.5/(r^1.5+a)
    Limite kepleriano newtoniano (a=0): -1.5 (Kato, Sez. 6.1).
    """
    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)
    return -1.5 * r**1.5 / (r**1.5 + a)


def growth_rate_p_mode(r, a, M, nu0, alpha, m=0, n=0, A=1.0):
    """
    G(r) per il p-mode fondamentale (m=0, n=0). NaN dove omega^2<=kappa(r)^2
    (fuori dalla finestra di propagazione).
    """
    r = np.asarray(r, dtype=float)
    kappa = 2 * np.pi * nu_r(r, a, M)      # rad/s
    Omega = 2 * np.pi * nu_phi(r, a, M)    # rad/s
    omega = 2 * np.pi * np.asarray(nu0, dtype=float)  # rad/s

    valid = omega**2 > kappa**2
    dlnOm = dlnOmega_dlnr(r, a)

    bracket = (Omega**2 / omega**2) * (A * dlnOm + kappa**2 / (2 * Omega**2)) + 2.0 / 3.0
    G = -alpha * (omega**2 - kappa**2) / Omega * bracket

    return np.where(valid, G, np.nan)


def find_p_mode_outer_boundary(a, M, nu0, m, n, r_scan_max=50.0, n_scan=8000):
    """
    r1: primo raggio (da ISCO verso l'esterno) dove kappa(r) = omega,
    bordo esterno della regione di trapping del p-mode fondamentale
    (Kato, Fig. 6, Sez. 5.1: regione intrappolata = [r_isco, r1]).
    Scalari (root-finding singolo).
    """
    a = float(a); M = float(M); nu0 = float(nu0)
    r_in = float(r_isco(a))

    r_scan = np.geomspace(r_in * 1.0001, r_scan_max, n_scan)
    kappa_scan = nu_r(r_scan, a, M)
    if m == 0:
        diff = nu0 - kappa_scan
        sign_changes = np.where(np.diff(np.sign(diff)) < 0)[0]
        if len(sign_changes) == 0:
            return np.nan

        i = sign_changes[0]
        denom = diff[i + 1] - diff[i]
        if denom == 0:
            return float(r_scan[i])
        r1 = r_scan[i] - diff[i] * (r_scan[i + 1] - r_scan[i]) / denom

    else:
        r1 = r_ilr(a, m, nu_obs=nu0, M=M)

    return float(r1)


##########################################################
# --------------------------------------------------------
##########################################################
# C-MODE: SMORZAMENTO COROTAZIONALE (Tsang & Lai 2008, arXiv:0810.1299)
#
# Notazione dell'articolo (Sez. 2-3, Eq. 10-13): Omega, Omega_perp,
# kappa in unita' c^3/GM; nel nostro codice usiamo le stesse grandezze
# fisiche gia' disponibili in setup.py (nu_phi, nu_theta, nu_r, tutte
# in Hz, con omega=2*pi*nu). Le equazioni (10)-(12) di Tsang & Lai
# coincidono esattamente con le nostre g_phi/g_theta/g_r gia' validate
# (stessa origine: Aliev & Gal'tsov 1981 / Okazaki et al. 1987), quindi
# non serve reimplementarle: si riusano nu_phi, nu_theta, nu_r.
#
# Geometria della finestra di trapping (Fig. 1-2 dell'articolo):
#   r_in (=r_isco) < r_IVR < r_ILR < r_c (corotazione)
#   - regione I  [r_in, r_IVR]  : zona di propagazione del c-mode
#     (trapped), condizione omega_tilde^2 > n*Omega_perp^2 > kappa^2
#   - regione II [r_IVR, r_ILR] : zona evanescente (tunneling),
#     condizione kappa^2 < omega_tilde^2 < n*Omega_perp^2
#   - oltre r_ILR: onda propagante verso la risonanza di corotazione,
#     dove viene assorbita (Sez. 4)
#
# Scelta di c_s: l'articolo usa una prescrizione propria c_s=beta*r*Omega_in
# (Sez. 5.3, illustrativa) SOLO per i risultati numerici di esempio; la
# formula (39) in se' e' generale e richiede soltanto UN profilo di
# c_s(r). Qui si riusa la prescrizione GIA' adottata in tutto il resto
# di questo progetto (disk_profiles.c_s_powerlaw, q=3/2, Motta et al.
# 2017) invece di introdurre una terza prescrizione di suono: scelta di
# coerenza interna del progetto, non del paper originale -- da
# dichiarare esplicitamente.
##########################################################


def _omega_tilde(r, a, M, m, nu0):
    """omega_tilde(r) = omega - m*Omega(r), rad/s."""
    omega = 2 * np.pi * np.asarray(nu0, dtype=float)
    Omega = 2 * np.pi * nu_phi(r, a, M)
    return omega - m * Omega


def _k_evanescent(r, a, M, m, n, nu0):
    """
    |k(r)| [cm^-1] nella zona evanescente (regione II, Eq. 13 di
    Tsang & Lai con il segno invertito sotto radice, dato che li'
    kappa^2<omega_tilde^2<n*Omega_perp^2):

        c_s^2 k^2 = (omega_tilde^2-kappa^2)*(n*Omega_perp^2-omega_tilde^2) / omega_tilde^2

    NaN dove la condizione di evanescenza non e' soddisfatta.
    """
    r = np.asarray(r, dtype=float)
    kappa = 2 * np.pi * nu_r(r, a, M)
    Omega_perp = 2 * np.pi * nu_theta(r, a, M)
    om_t = _omega_tilde(r, a, M, m, nu0)
    c_s = c_s_powerlaw(r, a)

    valid = (kappa**2 < om_t**2) & (om_t**2 < n * Omega_perp**2)
    num = (om_t**2 - kappa**2) * (n * Omega_perp**2 - om_t**2)
    with np.errstate(invalid='ignore'):
        k = np.sqrt(np.where(valid, num, np.nan)) / (c_s * np.abs(om_t))
    return k


def theta_II(a, M, m, n, nu0, r_IVR, r_ILR, n_rad=2000):
    """
    Theta_II = integrale_{r_IVR}^{r_ILR} |k| dr (Eq. 26 Tsang & Lai),
    l'esponente di soppressione per tunneling attraverso la zona
    evanescente. Adimensionale per costruzione (|k| in cm^-1, dr in cm).

    CORREZIONE DIMENSIONALE (stessa lezione del fattore Rg^2 in
    align_timescale.py): |k| e' gia' fisico (cm^-1), ma se si integra
    su una griglia radiale espressa in unita' di r_g (adimensionale,
    convenzione di tutto questo progetto), va moltiplicato un fattore
    Rg_cm = Rg_SUN*M per ottenere dr in cm.
    """
    from setup import Rg_SUN
    eps = 1e-9 * (r_ILR - r_IVR)
    x = np.linspace(r_IVR + eps, r_ILR - eps, n_rad)
    k = _k_evanescent(x, a, M, m, n, nu0)
    # scarta eventuali punti NaN residui (errore di floating point
    # vicinissimo ai bordi, dove uno dei due fattori sotto radice si
    # annulla per costruzione): piu' robusto di allargare l'epsilon a
    # tentativi, dato che l'errore residuo del root-finder che ha
    # trovato r_IVR/r_ILR non e' noto a priori.
    finite = np.isfinite(k)
    Rg_cm = Rg_SUN * M
    return np.trapezoid(k[finite], x[finite]) * Rg_cm


def _cmode_trapped_integrand(r, a, M, m, n, nu0):
    """
    Integrando fra parentesi quadre di Eq. (39) di Tsang & Lai,
    valutato nella regione I (trapping, [r_in, r_IVR]), SENZA il
    fattore Rg_cm (aggiunto separatamente in omega_i_cmode, stessa
    ragione di theta_II):

        |omega_tilde|^2 / sqrt[(kappa^2-omega_tilde^2)(n*Omega_perp^2-omega_tilde^2)]
        * (1 - n*kappa^2*Omega_perp^2/omega_tilde^4) / c_s(r)

    Nella regione I, omega_tilde^2 > n*Omega_perp^2 > kappa^2 (Sez. 3),
    quindi (kappa^2-omega_tilde^2) e (n*Omega_perp^2-omega_tilde^2) sono
    ENTRAMBI negativi: il prodotto sotto radice e' positivo per
    costruzione, nessun valore assoluto necessario.
    """
    r = np.asarray(r, dtype=float)
    kappa = 2 * np.pi * nu_r(r, a, M)
    Omega_perp = 2 * np.pi * nu_theta(r, a, M)
    om_t = _omega_tilde(r, a, M, m, nu0)
    c_s = c_s_powerlaw(r, a)

    with np.errstate(invalid='ignore'):
        denom = np.sqrt((kappa**2 - om_t**2) * (n * Omega_perp**2 - om_t**2))
        factor = 1.0 - n * kappa**2 * Omega_perp**2 / om_t**4
        result = (om_t**2 / denom) * factor / c_s
    return result


def find_r_IVR(a, M, m, n, nu0, r_scan_max=5000.0, n_scan=8000):
    """
    r_IVR: confine tra la Regione I (trapping, |omega_tilde|>sqrt(n)*Omega_perp
    >kappa) e la Regione II (evanescente, kappa<|omega_tilde|<sqrt(n)*Omega_perp),
    Tsang & Lai 2008 Sez. 3. Definito da |omega_tilde(r)|=sqrt(n)*Omega_perp(r).

    Wrapper diretto su setup.r_vertical_resonance: NESSUNA reimplementazione
    della logica di root-finding, che vive in un solo posto (setup.py,
    _find_zero_crossing) e viene qui solo richiamata con gli argomenti
    di questo modulo (m, n, nu0 al posto di nu_obs, M al posto di M_BH).
    """
    return r_vertical_resonance(a, m, n, nu_obs=nu0, M=M,
                                 n_scan=n_scan, r_scan_max=r_scan_max)


def find_r_ILR(a, M, m, n, nu0, r_IVR, r_scan_max=5000.0, n_scan=8000):
    """
    r_ILR: confine tra la Regione II (evanescente) e l'onda propagante
    verso la corotazione (Tsang & Lai 2008, Sez. 3-4). Definito da
    |omega_tilde(r)|=kappa(r) -- la risonanza di Lindblad interna
    standard, GIA' implementata in setup.r_ilr (usata altrove nel
    progetto per il modello RPM): la si richiama direttamente, senza
    duplicarne la logica.

    n non entra nella condizione dell'ILR (dipende solo da kappa, non
    da Omega_perp): resta in firma solo per uniformita' con
    find_r_IVR/omega_i_cmode (stessa lista di argomenti in tutta la
    pipeline del c-mode), non e' usato nel corpo della funzione.

    r_IVR e' richiesto in firma per coerenza con la pipeline di
    omega_i_cmode (che lo calcola e valida PRIMA di cercare r_ILR) e
    come promemoria dell'ordinamento fisico r_in<r_IVR<r_ILR<r_c gia'
    dichiarato nel docstring del modulo -- non serve pero' a restringere
    la scansione: la condizione |omega_tilde|=kappa non fa riferimento a
    Omega_perp, quindi il primo attraversamento neg->pos scandendo da
    r_isco (kappa(r_isco)=0 per definizione di ISCO, quindi si parte
    sempre da kappa<|omega_tilde|) individua univocamente r_ILR, senza
    rischio di intercettare per errore la soglia dell'IVR (che e' una
    condizione su una funzione diversa, Omega_perp, non su kappa).

    Se r_IVR non e' finito (nessuna Regione I), ritorna NaN
    immediatamente: la Regione II non e' definita senza un confine interno.
    """
    if not np.isfinite(r_IVR):
        return np.nan
    return r_ilr(a, m, nu_obs=nu0, M=M, n_scan=n_scan, r_scan_max=r_scan_max)


def omega_i_cmode(a, M, m, n, nu0, r_scan_max=5000.0, n_scan=4000, n_rad=2000):
    """
    Pipeline completa Eq. (39) di Tsang & Lai (2008): trova r_IVR,
    r_ILR, calcola Theta_II e l'integrale di trapping, restituisce
    omega_i [rad/s] (sempre <0, il c-mode e' sempre smorzato -- Sez.
    5.2, "the mode is always damped").

    Ritorna un dict con tutte le grandezze intermedie (utile per
    diagnostica/plot): r_in, r_IVR, r_ILR, theta_II, omega_i. Valori
    NaN se la finestra di trapping non esiste per questi (a,M,m,n,nu0).
    """
    a = float(a); M = float(M); nu0 = float(nu0)
    r_in = float(r_isco(a))

    r_IVR = find_r_IVR(a, M, m, n, nu0, r_scan_max, n_scan)
    if not np.isfinite(r_IVR):
        return dict(r_in=r_in, r_IVR=np.nan, r_ILR=np.nan,
                    theta_II=np.nan, omega_i=np.nan)

    r_ILR = find_r_ILR(a, M, m, n, nu0, r_IVR, r_scan_max, n_scan)
    if not np.isfinite(r_ILR):
        return dict(r_in=r_in, r_IVR=r_IVR, r_ILR=np.nan,
                    theta_II=np.nan, omega_i=np.nan)

    Th_II = theta_II(a, M, m, n, nu0, r_IVR, r_ILR, n_rad)

    from setup import Rg_SUN
    Rg_cm = Rg_SUN * M
    eps = 1e-9 * (r_IVR - r_in)
    x = np.linspace(r_in + eps, r_IVR - eps, n_rad)
    integrand = _cmode_trapped_integrand(x, a, M, m, n, nu0)
    finite = np.isfinite(integrand)
    I_hat = np.trapezoid(integrand[finite], x[finite])
    I_phys = I_hat * Rg_cm  # secondi

    omega_i = -0.25 * np.exp(-2 * Th_II) / I_phys

    return dict(r_in=r_in, r_IVR=r_IVR, r_ILR=r_ILR,
                theta_II=Th_II, omega_i=omega_i)


def t_damp_cmode(a, M, m, n, nu0, **kwargs):
    """t_damp = 1/|omega_i| [s], o NaN se la finestra non esiste."""
    res = omega_i_cmode(a, M, m, n, nu0, **kwargs)
    if not np.isfinite(res['omega_i']):
        return np.nan
    return 1.0 / abs(res['omega_i'])