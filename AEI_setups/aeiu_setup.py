# setup file globlae per AEI model
import numpy as np




def solve_k_aei(r_rg, a, B0, Sigma, c_s, m, M=M_BH):
    """
    Risolve la relazione di dispersione AEI (Tagger & Pellat 1999, Eq. 17):

        ω̃² = κ² + (2B₀²/Σ)(|k|/r) + (k²/r²) c_s²

    dove k è il numero d'onda nella variabile s = ln(r)  →  k adimensionale,
    e r è il raggio fisico in cm.

    ANALISI DIMENSIONALE:
      - ω̃², κ²       : s⁻²
      - 2B₀²/Σ · k/r : [G²/(g/cm²)] · [1/cm] = s⁻²  (r in cm, k adim.)
      - k²/r² · c_s² : [1/cm²] · [cm²/s²]    = s⁻²  (r in cm, k adim.)

    Il codice converte r [r_g] → r [cm] internamente prima di costruire
    i coefficienti. k restituito è adimensionale (numero d'onda in s=ln r).
    kperr = k (già adimensionale, equivale a k_fisico × r).

    Parameters
    ----------
    r_rg  : array_like   raggio in unità di r_g
    a     : float        spin adimensionale
    B0    : array_like   campo magnetico [G]
    Sigma : array_like   densità superficiale [g/cm²]
    c_s   : array_like   velocità del suono [cm/s]
    m     : int          numero d'onda azimutale
    M     : float        massa BH [M_sun]

    Returns
    -------
    k   : ndarray   numero d'onda adimensionale (in s = ln r)
                    NaN dove non esiste soluzione reale e positiva.
                    k_fisico [1/cm] = k / r_cm
                    kperr (adim.)   = k  (identicamente)
    """
    r_rg  = np.asarray(r_rg,  dtype=float)
    B0    = np.asarray(B0,    dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    c_s   = np.asarray(c_s,   dtype=float)

    # ── conversione r in cm ──────────────────────────────────────────────────
    Rg    = Rg_SUN * M          # cm per r_g
    r_cm  = r_rg * Rg           # raggio fisico in cm

    # ── frequenze ───────────────────────────────────────────────────────────
    kappa_sq  = (2 * np.pi * nu_r(r_rg,  a, M))**2   # s⁻²
    Omega_phi = 2 * np.pi * nu_phi(r_rg, a, M)        # s⁻¹
    omega     = 2 * np.pi * NU0                        # s⁻¹  (frequenza osservata)
    om_tilde  = omega - m * Omega_phi                  # s⁻¹

    # ── coefficienti quadratica  A k² + B k + CC = 0 ────────────────────────
    # r in cm → k adimensionale → unità omogenee s⁻²
    A  = c_s**2   / r_cm**2          # s⁻²
    B  = 2*B0**2  / (Sigma * r_cm)   # s⁻²
    CC = kappa_sq - om_tilde**2       # s⁻²

    Delta = B**2 - 4*A*CC

    k = np.full_like(r_rg, np.nan)
    good = Delta >= 0
    if np.any(good):
        sqD = np.sqrt(Delta[good])
        kp  = (-B[good] + sqD) / (2*A[good])
        km  = (-B[good] - sqD) / (2*A[good])
        # preferisci k_plus (più grande); se negativo prova k_minus
        k[good] = np.where(kp > 0, kp, np.where(km > 0, km, np.nan))

    return k   # adimensionale (numero d'onda in s = ln r)