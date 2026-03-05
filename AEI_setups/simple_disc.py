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


# ========================================================================
# SOLVER ANALITICO PER k
# ========================================================================

def solve_k_rossby(r, a, B00, Sigma0, alpha_B, alpha_S, m, hr, M=M_BH):
    """
    Solve for |k| from Rossby wave dispersion relation:
    
    (ω̃ + m·Ω_φ)² = κ² + (2B₀²/Σr)|k| + k²c_s²/r^2
    
    This is a quadratic in |k|:
    c_s²/r^2 × |k|² + (2B₀²/Σr) × |k| + (κ² - ω²) = 0
    
    where ω = ω̃ + m·Ω_φ
    
    Parameters:
    -----------
    r, a, B00, Sigma0, alpha_B, alpha_S : array_like
        Physical parameters (can be meshgrid arrays)
    m : int
        Azimuthal mode number (default: 1)
    hr : float
        Aspect ratio H/r (default: 0.05)
    
    Returns:
    --------
    k_solutions : dict
        'k_plus': larger solution
        'k_minus': smaller solution
        'valid': boolean mask where solutions are real and positive
    """
    r = np.asarray(r)
    a = np.asarray(a)
    B00 = np.asarray(B00)
    Sigma0 = np.asarray(Sigma0)
    alpha_B = np.asarray(alpha_B)
    alpha_S = np.asarray(alpha_S)
    
    # Compute profiles
    B0 = B0_profile(r, a, B00, alpha_B)
    Sigma = Sigma_profile(r, a, Sigma0, alpha_S)
    c_s = sound_speed_thin(r, a, hr, M)
    kappa_sq = (2 * np.pi * nu_r(r, a, M))**2  # rad²/s²
    
    # Target frequency in frame: ω = ω̃ + m·Ω_φ
    omega = 2 * np.pi * NU0  # rad/s (observed frequency)
    Omega_phi = 2 * np.pi * nu_phi(r, a, M)  # rad/s
    omega_tilde = omega - m * Omega_phi
    omega_sq = omega_tilde**2
    
    Rg = Rg_SUN * M  # cm
    r_cm = r * Rg
    
    # Quadratic coefficients: A|k|² + B|k| + C = 0
    A = c_s**2 / r**2
    B = 2 * B0**2 / Sigma / r
    C = kappa_sq - omega_sq
    
    # Discriminant
    Delta = B**2 - 4*A*C
    
    # Solutions
    k_plus = np.zeros_like(r)
    k_minus = np.zeros_like(r)
    valid_plus = np.zeros_like(r, dtype=bool)
    valid_minus = np.zeros_like(r, dtype=bool)
    
    # Only compute where discriminant is non-negative
    mask = Delta >= 0
    
    if np.any(mask):
        sqrt_Delta = np.sqrt(Delta[mask])
        k_plus[mask] = (-B[mask] + sqrt_Delta) / (2*A[mask])
        k_minus[mask] = (-B[mask] - sqrt_Delta) / (2*A[mask])
        
        # Keep only positive solutions
        valid_plus[mask] = (k_plus[mask] > 0)
        valid_minus[mask] = (k_minus[mask] > 0)
    
    return {
        'k_plus': k_plus,
        'k_minus': k_minus,
        'valid_plus': valid_plus,
        'valid_minus': valid_minus,
        'Delta': Delta,
        'B0': B0,
        'Sigma': Sigma,
        'c_s': c_s
    }


# ========================================================================
# FUNZIONI PER I CHECK FISICI
# ========================================================================

def check_k_physical(k, r, k_min_factor=0.1, k_max_factor=10):
    """
    Check if k is in physical range: k_min/r < |k| < k_max/r
    
    For thin disk with H ~ 0.05r, we expect k ~ 1/H ~ 20/r
    We allow a range [0.1/r, 10/r] to be conservative
    
    Returns:
    --------
    mask : boolean array
        True where k is physical
    """
    k = np.asarray(k)
    r = np.asarray(r)
    
    k_min = k_min_factor / r
    k_max = k_max_factor / r
    
    return (k >= k_min) & (k <= k_max)


def compute_beta(r, a, B00, Sigma0, alpha_B, alpha_S, hr, M=M_BH):
    """
    Compute plasma beta: β = 8π Σ c_s² / (H B₀²)
    
    For AEI instability, we need β ≤ 1 (magnetic pressure dominated)
    
    Returns:
    --------
    beta : array_like
        Plasma beta parameter
    """
    r = np.asarray(r)
    a = np.asarray(a)
    B00 = np.asarray(B00)
    Sigma0 = np.asarray(Sigma0)
    
    B0 = B0_profile(r, a, B00, alpha_B)
    Sigma = Sigma_profile(r, a, Sigma0, alpha_S)
    c_s = sound_speed_thin(r, a, hr, M)
    
    Rg = Rg_SUN * M  # cm
    H = hr * r * Rg  # cm
    
    beta = 8 * np.pi * Sigma * c_s**2 / (H * B0**2)
    
    return beta


def check_beta_AEI(r, a, B00, Sigma0, alpha_B, alpha_S, hr, M=M_BH, beta_max=1.0):
    """
    Check if β ≤ beta_max (condition for AEI instability)
    
    Returns:
    --------
    mask : boolean array
        True where β ≤ beta_max
    """
    beta = compute_beta(r, a, B00, Sigma0, alpha_B=alpha_B, alpha_S=alpha_S, hr=hr, M=M_BH)
    return beta <= beta_max


def compute_shear_quantity(r, a, B00, Sigma0, alpha_B, alpha_S, M=M_BH):
    """
    Compute Q = Ωφ × Σ / B₀²
    
    For AEI, we need dQ/dr > 0
    
    Returns:
    --------
    Q : array_like
        Shear quantity
    """
    r = np.asarray(r)
    a = np.asarray(a)
    B00 = np.asarray(B00)
    Sigma0 = np.asarray(Sigma0)
    
    # Compute profiles
    B0 = B0_profile(r, a, B00, alpha_B)
    Sigma = Sigma_profile(r, a, Sigma0, alpha_S)
    Omega_phi = 2 * np.pi * nu_phi(r, a, M)  # rad/s
    
    Q = Omega_phi * Sigma / B0**2
    
    return Q


def check_shear_positive(r, a, B00, Sigma0, alpha_B, alpha_S, M=M_BH, dr_factor=0.01):
    """
    Check if dQ/dr > 0 using finite differences
    
    Parameters:
    -----------
    dr_factor : float
        Fractional step size for finite difference: dr = dr_factor * r
    
    Returns:
    --------
    mask : boolean array
        True where dQ/dr > 0
    """
    r = np.asarray(r)
    a = np.asarray(a)
    B00 = np.asarray(B00)
    Sigma0 = np.asarray(Sigma0)
    
    # Compute Q at r and r + dr
    dr = dr_factor * r
    Q_r = compute_shear_quantity(r, a, B00, Sigma0, alpha_B, alpha_S, M)
    Q_rp = compute_shear_quantity(r + dr, a, B00, Sigma0, alpha_B, alpha_S, M)
    
    # Finite difference derivative
    dQ_dr = (Q_rp - Q_r) / dr
    
    return dQ_dr > 0


# ========================================================================
# FUNZIONE DI MATCHING CON OPZIONI PER I CHECK FISICI
# ========================================================================

def find_rossby_matches(param_dict, m, hr, alpha_B, alpha_S,
                       check_k=False, check_beta=False, check_shear=False, use_k_plus=True):
    """
    Find parameter combinations that produce Rossby waves at target frequency.
    
    Parameters:
    -----------
    param_dict : dict
        Parameter grid definition for create_param_grid()
        Must include: 'r', 'a', 'B00', 'Sigma0'
    use_k_plus : bool
        If True, use k_plus solution; if False, use k_minus
    check_k : bool
        If True, apply physical k constraint
    check_beta : bool
        If True, apply β ≤ 1 constraint
    check_shear : bool
        If True, apply dQ/dr > 0 constraint
    m : int
        Azimuthal mode number
    hr : float
        Aspect ratio H/r
    alpha_B, alpha_S : float
        Power-law exponent for B0 and Sigma profiles
    
    Returns:
    --------
    df : DataFrame
        Matched solutions with columns for all parameters plus 'k', 'beta', 'dQ_dr'
    """
    # Create parameter grid
    param_vectors, mesh_arrays = create_param_grid(param_dict, mesh=True)
    labels = list(param_dict.keys())
    
    # Extract meshgrid arrays
    param_mesh = {lab: arr for lab, arr in zip(labels, mesh_arrays)}
    
    r = param_mesh['r']
    a = param_mesh['a']
    B00 = param_mesh['B00']
    Sigma0 = param_mesh['Sigma0']
    
    # Solve for k
    k_sols = solve_k_rossby(r, a, B00, Sigma0, alpha_B=alpha_B, alpha_S=alpha_S, m=m, hr=hr)
    
    # Choose which solution to use
    k = k_sols['k_plus'] if use_k_plus else k_sols['k_minus']
    
    # Build mask
    mask = k_sols['valid_plus'] if use_k_plus else k_sols['valid_minus']
    
    # Apply ISCO constraint
    a_vec = param_vectors['a']
    isco = r_isco(a_vec)
    r_isco_nd = isco.reshape(-1, *[1]*(r.ndim - 1))
    mask &= (r >= r_isco_nd)
    
    # Collect results
    Rg = Rg_SUN * M_BH  # cm to convert k from 1/cm to 1/rg
    k = k * Rg  # convert k from 1/cm to 1/rg
    
    # Compute additional quantities for all points
    beta = compute_beta(r, a, B00, Sigma0, alpha_B=alpha_B, alpha_S=alpha_S, hr=hr)
    Q = compute_shear_quantity(r, a, B00, Sigma0, alpha_B=alpha_B, alpha_S=alpha_S)
    
    # Compute derivative of Q
    dr_factor = 0.01
    dr = dr_factor * r
    Q_rp = compute_shear_quantity(r + dr, a, B00, Sigma0, alpha_B=alpha_B, alpha_S=alpha_S)
    dQ_dr = (Q_rp - Q) / dr
    
    # Apply physical constraints if requested
    if check_k:
        mask &= check_k_physical(k, r)
    
    if check_beta:
        mask &= check_beta_AEI(r, a, B00, Sigma0, alpha_B=alpha_B, alpha_S=alpha_S, hr=hr)
    
    if check_shear:
        mask &= check_shear_positive(r, a, B00, Sigma0, alpha_B=alpha_B, alpha_S=alpha_S)
    
    rows = []
    idxs = np.argwhere(mask)
    
    for idx in idxs:
        idx_tuple = tuple(idx)
        row = {lab: arr[idx_tuple] for lab, arr in param_mesh.items()}
        row['k'] = k[idx_tuple]
        row['kperr'] = k[idx_tuple] * r[idx_tuple]
        row['beta'] = beta[idx_tuple]
        row['dQ_dr'] = dQ_dr[idx_tuple]
        row['alpha_B'] = alpha_B
        row['alpha_S'] = alpha_S
        row['m'] = m
        row['hr'] = hr
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df