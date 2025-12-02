import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import LogNorm

from setup import *

# --------------------------------------------------------
# FUNCTIONS TO PLOT FREQUENCY LEVEL CURVES
def spin_double(X, Y, F, spin_on_x, spin_on_y):
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
        sort_idx = np.argsort(X[0, :])
        X = X[:, sort_idx]
        Y = Y[:, sort_idx]
        F = F[:, sort_idx]
    else:
        X = np.concatenate([other, other], axis=0)
        Y = np.concatenate([A_neg, A], axis=0)
        F = np.concatenate([F_neg, F], axis=0)
        sort_idx = np.argsort(Y[:, 0])
        X = X[sort_idx, :]
        Y = Y[sort_idx, :]
        F = F[sort_idx, :]
    
    return X, Y, F

def mask_radii(x_param, y_param, X, Y, F, mesh_arrays, labels, idx_fix):
    mask = np.ones_like(F, dtype=bool)

    if x_param == "a" or y_param == "a":
        if y_param == "a":
            R = X
            A = Y
        else:
            R = Y
            A = X
        
        a_vals = np.unique(A)
        risco_vals = r_isco(a_vals)
        risco_map = dict(zip(a_vals, risco_vals))
        risco_grid = np.vectorize(risco_map.get)(A)
        mask &= (R >= risco_grid)

    else:
        spin_param = "a"
        try:
            ia = labels.index(spin_param)
        except ValueError:
            pass

        a_vals = mesh_arrays[ia][:, 0, 0]
        a_fixed = a_vals[idx_fix]
        R_ISCO_FIXED = r_isco(a_fixed)
        
        # Controlla se r è sull'asse X o Y
        if x_param == "r":
            R_GRID = X
        elif y_param == "r":
            R_GRID = Y
        else:
            # Non dovrebbe succedere se siamo in questo blocco, ma per sicurezza
            R_GRID = None 
            
        if R_GRID is not None:
            # La condizione fisica è che il raggio deve essere >= r_isco
            mask = (R_GRID >= R_ISCO_FIXED)

    return mask


def add_isco(x_param, y_param, X, Y, mesh_arrays, labels, idx_fix):
    # Estrai vettore 'a' e vettore 'r' dall'asse corretto
    if x_param == "a":
        a_vals = np.unique(X)
        # compute risco(a)
        risco_vals = np.array([r_isco(a) for a in a_vals])
        # plot on XY-plane
        plt.plot(a_vals, risco_vals, "--", color="purple", lw=2, label="ISCO")
    
    elif y_param == "a":
        a_vals = np.unique(Y)
        risco_vals = np.array([r_isco(a) for a in a_vals])
        plt.plot(risco_vals, a_vals, "--", color="purple", lw=2, label="ISCO")

    else:
        spin_param = "a"
        try:
            ia = labels.index(spin_param)
        except ValueError:
            # Se 'a' non è tra le etichette, non facciamo nulla.
            pass
        
        a_vals = mesh_arrays[ia][:, 0, 0]
        a_fixed = a_vals[idx_fix]

        if x_param == "r":
            y_vals = np.unique(Y)
            isco = np.ones_like(y_vals) * r_isco(a_fixed)
            isco_neg = np.ones_like(y_vals) * r_isco(-a_fixed)
            plt.plot(isco, y_vals, "--", color="purple", lw=2, label="ISCO")
            plt.plot(isco_neg, y_vals, ":", color="purple", lw=2, label="ISCO (-a)")
        elif y_param == "r":
            x_vals = np.unique(X)
            isco = np.ones_like(x_vals) * r_isco(a_fixed)
            isco_neg= np.ones_like(x_vals) * r_isco(-a_fixed)
            plt.plot(x_vals, isco, "--", color="purple", lw=2, label="ISCO")
            plt.plot(x_vals, isco_neg, ":", color="purple", lw=2, label="ISCO (-a)")

def plot_param_colormap(mesh_arrays, labels, freq_grid,
                        x_param, y_param, idx_fix=-1,
                        title=None,
                        colormap="inferno",
                        log_x=False, log_y=False,
                        apply_isco=True,
                        add_target=True):
    """
    Versione colormap di plot_param_contour.
    Mantiene:
      - slicing automatico N-D
      - duplicazione ±a
      - maschera ISCO
      - target frequency
      - scaling log
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
    slicer = [idx_fix] * ndim

    for j in range(ndim):
        if slicer[j] >= mesh_arrays[j].shape[j]:
            slicer[j] = -1    # ultimo elemento dell'asse

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
        X, Y, F = spin_double(X, Y, F, spin_on_x, spin_on_y)

    # -----------------------------------------
    # 3) MASCHERA ISCO
    # -----------------------------------------
    mask = np.ones_like(F, dtype=bool)
    if apply_isco and ("r" in x_param or "r" in y_param):
        mask = mask_radii(x_param, y_param, X, Y, F, mesh_arrays, labels, idx_fix)

    F_masked = np.ma.masked_where(~mask, F)

    # -----------------------------------------
    # 4) COLORMAP (pcolormesh)
    # -----------------------------------------

    pcm = plt.pcolormesh(
        X, Y, F_masked,
        shading="auto",
        cmap=colormap,
        norm=LogNorm()
    )
    cbar = plt.colorbar(pcm, label="frequency")

    # -----------------------------------------
    # 5) Target frequency NU0 (opzionale)
    # -----------------------------------------
    if add_target:
        plt.contour(
            X, Y, F_masked,
            levels=[TARGET_MIN, NU0, TARGET_MAX],
            colors=["red", "green", "darkblue"],
            linewidths=[2, 4, 2]
        )
    # Aggiungi il handle manualmente
    target_handle = plt.Line2D([], [], color="green", linewidth=4, label=f"Target ({NU0} Hz)")
    min_habdle = plt.Line2D([], [], color="red", linewidth=2, label=f"{TARGET_MIN} Hz")
    max_habdle = plt.Line2D([], [], color="darkblue", linewidth=2, label=f"{TARGET_MAX} Hz")
    isco_handle = plt.Line2D([], [], color="purple", linestyle="--", label="ISCO")


    # -----------------------------------------
    # 6) Aggiungi curva ISCO (se r vs a)
    # -----------------------------------------
    if apply_isco and ("r" in y_param or "r" in x_param):
        add_isco(x_param, y_param, X, Y, mesh_arrays, labels, idx_fix)
        if("a" not in y_param and "a" not in x_param):
            isco_neg_handle = plt.Line2D([], [], color="purple", linestyle=":", label="ISCO (-a)")
        else:
            isco_neg_handle = plt.Line2D([], [], color="purple", linestyle=":", label="")

    # -----------------------------------------
    # 7) Scaling assi
    # -----------------------------------------
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.legend(handles=[target_handle, min_habdle, max_habdle, isco_handle, isco_neg_handle], loc="upper right")
    plt.title(title or f"Colormap of frequency vs {x_param}, {y_param}")
    plt.xlabel(x_param)
    plt.ylabel(y_param)