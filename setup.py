# file for basic functions and parameters for various models
import numpy as np
import pandas as pd
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
def find_matches(mesh_arrays, labels, param_vectors, frq_fun):
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

    # Parametri N-D (meshgrid)
    param_dict = {lab: arr for lab, arr in zip(labels, mesh_arrays)}

    # Frequenze sul reticolo
    freq_func = frq_wrap(frq_fun)
    freq = freq_func(param_dict)

    # Maschera del match in frequenza
    mask_freq = np.abs(freq - NU0) < TOL

    # ------- rISCO positive & negative -------
    a_vec = param_vectors["a"]          # 1D array
    r_isco_pos = r_isco(a_vec)
    r_isco_neg = r_isco(-a_vec)

    # broadcasting
    r_isco_pos_nd = r_isco_pos.reshape(-1, *[1]*(freq.ndim - 1))
    r_isco_neg_nd = r_isco_neg.reshape(-1, *[1]*(freq.ndim - 1))

    # maschere complete
    mask_pos = np.ones_like(freq, bool)
    mask_neg = np.ones_like(freq, bool)

    # applica il vincolo solo ai parametri che contengono "r"
    for lab, arr in param_dict.items():
        if "r" in lab:
            mask_pos &= (arr >= r_isco_pos_nd)
            mask_neg &= (arr >= r_isco_neg_nd)

    # maschere finali
    mask_match_pos = mask_freq & mask_pos
    mask_match_neg = mask_freq & mask_neg

    # ------- raccogli risultati -------
    rows = []

    # POSITIVE SPIN (a rimane positivo)
    idxs_pos = np.argwhere(mask_match_pos)
    for idx in idxs_pos:
        row = {lab: arr[tuple(idx)] for lab, arr in param_dict.items()}
        row["freq"] = freq[tuple(idx)]
        rows.append(row)

    # NEGATIVE SPIN (invertiamo il segno di 'a')
    idxs_neg = np.argwhere(mask_match_neg)
    for idx in idxs_neg:
        row = {lab: arr[tuple(idx)] for lab, arr in param_dict.items()}
        row["a"] = -row["a"]     # ⬅⬅ salva direttamente il valore negativo
        row["freq"] = freq[tuple(idx)]
        rows.append(row)

    # ---- DataFrame finale ----
    df = pd.DataFrame(rows)
    return df


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
#FUNCTION TO PLOT FREQUENCY LEVEL CURVES
def plot_param_contour(mesh_arrays, labels, freq_grid, x_param, y_param, title=None, 
    levels=None, colormap="inferno", log_x=False, log_y=False, apply_isco=True,):
    """
    Generic contour plot from N-dimensional parameter grids.
    Extracts a 2D slice along (x_param, y_param).
    If spin a is one of the axes, automatically adds the mirrored (-a) branch.
    """

    # -----------------------------------------
    # 1) IDENTIFICAZIONE ASSI SELEZIONATI
    # -----------------------------------------
    try:
        ix = labels.index(x_param)
        iy = labels.index(y_param)
    except ValueError:
        raise ValueError("x_param or y_param not in labels")

    ndim = len(labels)
    all_axes = list(range(ndim))

    # Assi da bloccare (tutti gli altri)
    frozen_axes = [ax for ax in all_axes if ax not in (ix, iy)]

    # Slice: usiamo 0 per assi congelati, slice(None) per i due assi selezionati
    slicer = [0] * ndim
    slicer[ix] = slice(None)
    slicer[iy] = slice(None)
    slicer = tuple(slicer)

    # Estraggo la griglia 2D
    X = mesh_arrays[ix][slicer]
    Y = mesh_arrays[iy][slicer]
    F = freq_grid[slicer]

    # -----------------------------------------
    # 2) SE a È UNO DEI DUE ASSI → DUPLICA GRIGLIA
    # -----------------------------------------
    spin_on_x = (x_param == "a")
    spin_on_y = (y_param == "a")

    if spin_on_x or spin_on_y:
        # Identifica dove sta A e dove sta R
        if spin_on_x:
            A = X
            other = Y
        else:
            A = Y
            other = X

        # Parte negativa a -> -a
        A_neg = -A

        # Riutilizziamo però esattamente la stessa frequenza (simmetria),
        # ma la ISCO sarà diversa e la maschera finale lo rifletterà.
        F_neg = F.copy()

        # Ricombino in un’unica griglia (asse spin raddoppiato)
        if spin_on_x:
            X = np.concatenate([A_neg, A], axis=1)
            Y = np.concatenate([other, other], axis=1)
            F = np.concatenate([F_neg, F], axis=1)
        else:
            X = np.concatenate([other, other], axis=0)
            Y = np.concatenate([A_neg, A], axis=0)
            F = np.concatenate([F_neg, F], axis=0)

    # -----------------------------------------
    # 3) MASCHERA ISCO
    # -----------------------------------------
    mask = np.ones_like(F, dtype=bool)

    if apply_isco and (
        ("r" in x_param and "a" in y_param) or ("a" in x_param and "r" in y_param)
    ):
        if "a" in x_param:
            A = X
            R = Y
        else:
            A = Y
            R = X

        # Calcolo ISCO per ogni spin (positivo o negativo)
        a_vals = np.unique(A)
        risco_vals = np.array([r_isco(a) for a in a_vals])
        risco_map = dict(zip(a_vals, risco_vals))

        risco_grid = np.vectorize(risco_map.get)(A)
        mask &= (R >= risco_grid)

    F_masked = np.ma.masked_where(~mask, F)

    # -----------------------------------------
    # 4) CONTOUR
    # -----------------------------------------
    plt.figure(figsize=(9, 6))

    if levels is None:
        valid = F_masked[F_masked > 0]
        levels = np.logspace(np.log10(valid.min()),
                             np.log10(valid.max()), 25)

    cs = plt.contour(X, Y, F_masked, levels=levels, cmap=colormap)
    plt.colorbar(cs, label="frequency")

    # Highlight target band
    plt.contour(X, Y, F_masked, levels=[NU0-10*TOL], colors='red', linewidths=5)
    plt.contour(X, Y, F_masked, levels=[NU0+10*TOL], colors='blue', linewidths=5)

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(title or f"Contour of frequency vs {x_param}, {y_param}")
    plt.tight_layout()
    plt.show()


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