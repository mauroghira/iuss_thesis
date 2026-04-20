import sys
sys.path.append("..")
from setup import *

# ========================================================================
# PROFILI RADIALI
# ========================================================================

def B0_profile(r, a, B00, alpha_exp):
    """
    Campo magnetico: B₀(r) = B₀₀ × (r/r_in)^(-α)
    
    Parameters:
    -----------
    r : array_like
        Radius in gravitational radii
    a : array_like
        Dimensionless spin
    B00 : float
        Normalization at r_H
    alpha_exp : float
        Power-law exponent
    
    Returns:
    --------
    B0 : array_like
        Magnetic field (arbitrary units)
    """
    r = np.asarray(r)
    a = np.asarray(a)
    
    r_in = r_horizon(a)
    return B00 * (r / r_in)**(-alpha_exp)


def Sigma_profile(r, a, Sigma0, alpha_exp):
    """
    Surface density: Σ(r) = Σ₀ × (r/r_in)^(-α)
    
    Returns:
    --------
    Sigma : array_like
        Surface density (arbitrary units)
    """
    r = np.asarray(r)
    a = np.asarray(a)
    
    r_in = r_isco(a)
    return Sigma0 * (r / r_in)**(-alpha_exp)


def sound_speed_thin(r, a, hr=0.05, M=M_BH):
    """
    Sound speed in thin disk: c_s ≈ (H/r) × v_φ = (H/r) × r × Ω_φ
    
    Parameters:
    -----------
    hr : float
        Aspect ratio H/r (default: 0.05)
    
    Returns:
    --------
    c_s : array_like
        Sound speed in cm/s
    """
    r = np.asarray(r)
    a = np.asarray(a)
    M = np.asarray(M)
    
    # v_φ = r × Ω_φ = r × (2π ν_φ)
    # Ma ν_φ è già in Hz, quindi v_φ in cm/s è:
    Rg = Rg_SUN * M  # cm
    v_phi = 2 * np.pi * nu_phi(r, a, M) * r * Rg  # cm/s
    
    return hr * v_phi


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ADAPTER PER FZ UNIVERSASLI
# ═══════════════════════════════════════════════════════════════════════════════
def disk_model_simple(r_rg, a, B00, Sigma0, alpha_B, alpha_S, hr, M=M_BH):
    """
    Modello di disco semplificato con leggi di potenza per B₀ e Σ.

    Firma di ritorno allineata con SS/NT:
        B0, Sigma, c_s, hr_arr, zone, info

    dove hr_arr è costante = hr (H/r fisso per tutto il disco).
    zone è un array di 'N/A' (nessuna struttura a zone).
    info contiene i parametri usati (B00, Sigma0, alpha_B, alpha_S).
    """
    r_rg  = np.asarray(r_rg, float)
    B0    = B0_profile(r_rg, a, B00, alpha_B)
    Sigma = Sigma_profile(r_rg, a, Sigma0, alpha_S)
    c_s   = sound_speed_thin(r_rg, a, hr, M)
    hr_arr = np.full(len(r_rg), float(hr))
    zone   = np.full(len(r_rg), 'N/A', dtype=object)
    info   = {
        'B00':     B00,
        'Sigma0':  Sigma0,
        'alpha_B': alpha_B,
        'alpha_S': alpha_S,
    }
    return B0, Sigma, c_s, hr_arr, zone, info