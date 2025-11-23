# file for basic functions and parameters for various models
import numpy as np
import inspect

#let's set the parameters
M_BH = 10**6.3
NU0 = 3.3*10**(-5)

# Physical constants
G = 6.67430e-11       # m^3 kg^-1 s^-2
C = 299792458         # m/s
MSUN = 1.98847e30     # kg

Rg_SUN = G * MSUN / C**2  # in m

#let's set the tolerance for the target frequency
TOL = 0.01 * NU0  # 1% tolerance
TARGET_MIN = 1e-5
TARGET_MAX = 1e-4


# --------------------------------------------------------
# KERR METRIC BASIC FUNCTIONS
# Kerr frequencies in Hz
# i compute Rg separating M because I'll may try some different value
def nu_phi(r, a, M=M_BH):
    r = np.asarray(r)
    a = np.asarray(a)
    M = np.asarray(M)

    # r in units of GM/c^2, a dimensionless (0–1), M in solar masses
    Rg = Rg_SUN * M  # in cm
    return C / (2*np.pi * Rg*(r**1.5 + abs(a)))

def nu_theta(r, a, M=M_BH):
    r = np.asarray(r)
    a = np.asarray(a)
    vphi = nu_phi(r, a, M)
    factor = np.sqrt(1 - (4*abs(a))/r**1.5 + (3*a**2)/r**2)
    return vphi * factor

def nu_r(r, a, M=M_BH):
    r = np.asarray(r)
    a = np.asarray(a)
    vphi = nu_phi(r, a, M)
    factor = np.sqrt(1 - (6)/r + (8*abs(a))/r**1.5 - (3*a**2)/r**2)
    return vphi * factor

# Kerr ISCO radius
def r_isco(a):
    a = np.asarray(a)
    # sign(a) but safe for vectorization (returns 0 if a=0)
    sgn = np.sign(a)

    Z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
    Z2 = np.sqrt(3*a**2 + Z1**2)
    return 3 + Z2 - sgn*np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2))


# --------------------------------------------------------
# GENERAL FUNCTION TO CREATE GRIDS
def create_param_grid(param_dict, mesh=True, flatten=False):
    """
    Create an N-dimensional grid from a dictionary of parameters.
    
    param_dict : dict
        Keys = parameter names (str)
        Values = tuple/list defining the grid for each parameter.
                 Accepted formats:
                    - (min, max, n_points)
                    - array-like explicit list of values
                    
    mesh : bool
        If True, returns a meshgrid (N arrays).
        If False, only returns the 1D vectors for each parameter.

    flatten : bool
        If True, also returns a 2D array shape (N_points_total, N_params)
        useful for vectorized evaluation.

    Returns:
        param_vectors: dict of 1D arrays for each parameter
        mesh_arrays  : list of meshgrid arrays (if mesh=True)
        flat_matrix  : 2D array of flattened grid points (if flatten=True)
    """

    labels = list(param_dict.keys())
    values = []

    # Build 1D arrays for each parameter
    for key, val in param_dict.items():
        if len(val) == 3 and all(isinstance(x, (int, float)) for x in val):
            vmin, vmax, n = val
            values.append(np.linspace(vmin, vmax, n))
        else:
            # explicit array provided
            values.append(np.array(val))

    # Return only vectors
    param_vectors = {lab: vec for lab, vec in zip(labels, values)}

    if not mesh and not flatten:
        return param_vectors

    # Build meshgrid
    mesh_arrays = np.meshgrid(*values, indexing="ij")

    if not flatten:
        return param_vectors, mesh_arrays

    # Flatten meshgrid to shape (N_total, N_params)
    stacked = np.stack(mesh_arrays, axis=-1)   # shape (..., N_params)
    flat_matrix = stacked.reshape(-1, len(labels))

    return param_vectors, mesh_arrays, flat_matrix


# --------------------------------------------------------
# GENERAL FUNCTION TO FIND MATCHES
def find_param_matches(mesh_arrays, labels, param_vectors, freq_func):
    """
    N-dimensional match finder with automatic rISCO constraints.
    
    mesh_arrays : list of ndarrays
        Meshgrid arrays from create_param_grid()
    labels : list of str
        Parameter names (same order as mesh_arrays)
    param_vectors : dict
        1D parameter vectors (output of create_param_grid)
    r_isco : callable
        Function r_isco(a)
    freq_func : callable
        freq = freq_func(param_dict)
    """

    # Assemble parameter dictionary (each is an N-D meshgrid array)
    param_dict = {lab: arr for lab, arr in zip(labels, mesh_arrays)}

    # ---- Compute frequency on full grid ----
    freq = freq_func(param_dict)

    # ---- Master mask ----
    mask = np.ones_like(freq, dtype=bool)

    # ---- Automatic rISCO constraint ----
    # 1) Get the 1D vector of a-values
    a_vec = param_vectors["a"]                  # shape (N_a,)
    r_isco_vec = r_isco(a_vec)                  # shape (N_a,)

    # 2) Create broadcastable version
    r_isco_nd = r_isco_vec.reshape(-1, *[1]*(freq.ndim - 1))

    # 3) Apply to all r-parameters
    for lab, arr in param_dict.items():
        if "r" in lab:                          # <-- automatic detection
            mask &= arr >= r_isco_nd

    # ---- Apply frequency matching ----
    mask &= np.abs(freq - NU0) < TOL

    # ---- Collect matches ----
    idxs = np.argwhere(mask)   # shape (N_matches, ndim)
    results = []

    for idx in idxs:
        params = {lab: arr[tuple(idx)] for lab, arr in param_dict.items()}

        results.append({
            "params": params,
            "freq": freq[tuple(idx)]
        })

    return results

# --------------------------------------------------------
# GENERAL WRAPPER TO MAKE FREQUENCY FUNCTIONS
def frq_wrap(freq_callable):
    """
    Ritorna una funzione che accetta un dizionario di parametri
    e passa alla freq_callable solo quelli che essa richiede.
    """
    sig = inspect.signature(freq_callable)
    param_names = list(sig.parameters.keys())

    def wrapper(param_dict):
        # Estrai solo i parametri richiesti
        args = [param_dict[name] for name in param_names]
        return freq_callable(*args)

    return wrapper


# --------------------------------------------------------
# RELATIVISTIC PRECESSION MODEL FREQUENCIES
# lense-thirring precession frequency
def nu_LT(r, a, M=M_BH):
    return nu_phi(r, a, M) - nu_theta(r, a, M)

#periastron precession frequency
def nu_per(r, a, M=M_BH):
    return nu_phi(r, a, M) - nu_r(r, a, M)


# --------------------------------------------------------
# PRECESSING INNER FLOW MODEL FREQUENCY
def nu_solid_vect(a, rin, rout, zeta, n_rad=2000):
    a    = np.atleast_1d(a)
    rin  = np.atleast_1d(rin)
    rout = np.atleast_1d(rout)
    zeta = np.atleast_1d(zeta)

    x = np.linspace(0, 1, n_rad)
    x = x.reshape((1,) * a.ndim + (n_rad,))

    R = rin[..., None] + x * (rout - rin)[..., None]

    Sigma  = R**(-zeta[..., None])
    weight = Sigma * R**3 * nu_phi(R, a[..., None])

    num = np.trapezoid(nu_LT(R, a[..., None]) * weight, x, axis=-1)
    den = np.trapezoid(weight, x, axis=-1)

    return (num / den).squeeze()