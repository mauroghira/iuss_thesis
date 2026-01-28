import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from setup import *

def c_frq(m, j, a, r, M=M_BH):
    r = np.asarray(r)
    a = np.asarray(a)
    M = np.asarray(M)
    j = np.asarray(j)
    m = np.asarray(m)

    return m * nu_phi(r, a, M) - np.sqrt(j) * nu_theta(r, a, M)

# --------------------------------------------------------
# APPROCCIO GRAFICO: TROVA RIVR PER INTERPOLAZIONE
# --------------------------------------------------------

def find_rivr_graphical(r_grid, freq_grid, nu_target, method='linear'):
    """
    Trova rivr dove freq_grid incrocia nu_target usando interpolazione.
    
    Parameters:
    -----------
    r_grid : array
        Griglia dei raggi (1D o broadcast-compatible con freq_grid)
    freq_grid : array
        Frequenze calcolate sulla griglia
    nu_target : float
        Frequenza target da trovare
    method : str
        'linear' o 'cubic' per interpolazione
    
    Returns:
    --------
    rivr : array
        Raggi dove freq = nu_target (NaN dove non trova soluzione)
    """
    from scipy.interpolate import interp1d
    
    # Assicurati che r_grid sia 1D lungo l'ultimo asse
    r_1d = np.asarray(r_grid)
    freq = np.asarray(freq_grid)
    
    # Inizializza risultato
    rivr_shape = freq.shape[:-1]  # tutti gli assi tranne l'ultimo
    rivr = np.full(rivr_shape, np.nan)
    
    # Itera su tutti i punti tranne l'asse dei raggi
    for idx in np.ndindex(rivr_shape):
        r_slice = r_1d if r_1d.ndim == 1 else r_1d[idx + (slice(None),)]
        f_slice = freq[idx + (slice(None),)]
        
        # Maschera valori finiti e positivi
        mask = np.isfinite(f_slice) & (f_slice > 0) & np.isfinite(r_slice)
        
        if np.sum(mask) < 2:
            continue
        
        r_valid = r_slice[mask]
        f_valid = f_slice[mask]
        
        # Ordina per r crescente
        sort_idx = np.argsort(r_valid)
        r_valid = r_valid[sort_idx]
        f_valid = f_valid[sort_idx]
        
        # Controlla se nu_target è nel range
        if f_valid.min() <= nu_target <= f_valid.max():
            try:
                # Interpola
                interp_func = interp1d(f_valid, r_valid, kind=method, 
                                      bounds_error=False, fill_value=np.nan)
                rivr[idx] = interp_func(nu_target)
            except:
                pass
    
    return rivr


# --------------------------------------------------------
# VISUALIZZAZIONE INTERATTIVA CON RIVR
# --------------------------------------------------------

def plot_freq_with_rivr(a_vals, r_vals, freq_grid, m_vals, j_vals, 
                        nu_target=NU0, M_val=M_BH,
                        figsize=(16, 10)):
    """
    Plotta frequenze vs spin con identificazione grafica di rivr.
    
    Parameters:
    -----------
    a_vals : array (n_a,)
    r_vals : array (n_r,)
    freq_grid : array shape (n_m, n_j, n_a, n_r)
    m_vals : array (n_m,)
    j_vals : array (n_j,)
    """
    
    n_m = len(m_vals)
    n_j = len(j_vals)
    
    fig, axes = plt.subplots(n_m, n_j, figsize=figsize, squeeze=False)
    
    for im, mm in enumerate(m_vals):
        for ij, jj in enumerate(j_vals):
            ax = axes[im, ij]
            
            # Per ogni valore di 'a', plotta freq vs r
            for ia, aa in enumerate(a_vals[::10]):  # subsample per leggibilità
                freq_slice = freq_grid[im, ij, ia*10, :]
                
                mask = np.isfinite(freq_slice) & (freq_slice > 0)
                
                if np.sum(mask) > 0:
                    ax.plot(r_vals[mask], freq_slice[mask], 
                           alpha=0.3, color='gray', linewidth=0.5)
            
            # Trova rivr per ogni 'a'
            rivr_vals = []
            a_with_solution = []
            
            for ia, aa in enumerate(a_vals):
                freq_at_a = freq_grid[im, ij, ia, :]
                mask = np.isfinite(freq_at_a) & (freq_at_a > 0)
                
                if np.sum(mask) < 2:
                    continue
                
                r_valid = r_vals[mask]
                f_valid = freq_at_a[mask]
                
                # Ordina
                sort_idx = np.argsort(r_valid)
                r_valid = r_valid[sort_idx]
                f_valid = f_valid[sort_idx]
                
                # Trova dove freq = nu_target
                if f_valid.min() <= nu_target <= f_valid.max():
                    from scipy.interpolate import interp1d
                    try:
                        interp_func = interp1d(f_valid, r_valid, kind='linear')
                        rivr = interp_func(nu_target)
                        rivr_vals.append(rivr)
                        a_with_solution.append(aa)
                    except:
                        pass
            
            # Plotta rivr vs a
            if len(rivr_vals) > 0:
                ax.plot(rivr_vals, [nu_target]*len(rivr_vals), 
                       'ro', markersize=4, label=f'rivr(a) at ν={nu_target:.2e}')
            
            # Target frequency
            ax.axhline(nu_target, color='green', linestyle='--', 
                      linewidth=2, label=f'Target ({nu_target:.2e} Hz)')
            ax.fill_between(r_vals, TARGET_MIN, TARGET_MAX, 
                          color='gray', alpha=0.2, label='Target range')
            
            # ISCO per a=0 (esempio)
            r_isco_mid = r_isco(0)
            ax.axvline(r_isco_mid, color='purple', linestyle=':', 
                      linewidth=1.5, label=f'ISCO (a=0)')
            
            ax.set_xlabel('r [Rg]')
            ax.set_ylabel('ν [Hz]')
            ax.set_title(f'm={mm:.1f}, j={jj:.1f}')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best')
    
    plt.suptitle(f'Frequency vs radius (M = {M_val:.2e} Msun)', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    
    return fig


# --------------------------------------------------------
# PLOT 2D: RIVR vs (a, m, j) - HEATMAP
# --------------------------------------------------------

def plot_rivr_heatmap(a_vals, m_vals, j_vals, r_vals, freq_grid, 
                      nu_target=NU0, M_val=M_BH):
    """
    Crea heatmap di rivr in funzione dei parametri.
    """
    
    n_m = len(m_vals)
    n_j = len(j_vals)
    
    # Calcola rivr per tutta la griglia
    rivr_grid = np.full((n_m, n_j, len(a_vals)), np.nan)
    
    for im in range(n_m):
        for ij in range(n_j):
            for ia in range(len(a_vals)):
                freq_slice = freq_grid[im, ij, ia, :]
                
                mask = np.isfinite(freq_slice) & (freq_slice > 0)
                if np.sum(mask) < 2:
                    continue
                
                r_valid = r_vals[mask]
                f_valid = freq_slice[mask]
                
                sort_idx = np.argsort(r_valid)
                r_valid = r_valid[sort_idx]
                f_valid = f_valid[sort_idx]
                
                if f_valid.min() <= nu_target <= f_valid.max():
                    from scipy.interpolate import interp1d
                    try:
                        interp_func = interp1d(f_valid, r_valid, kind='linear')
                        rivr_grid[im, ij, ia] = interp_func(nu_target)
                    except:
                        pass
    
    # Plot
    fig, axes = plt.subplots(1, n_m, figsize=(5*n_m, 5), squeeze=False)
    axes = axes.flatten()
    
    for im, mm in enumerate(m_vals):
        ax = axes[im]
        
        # Crea meshgrid per j e a
        A_mesh, J_mesh = np.meshgrid(a_vals, j_vals, indexing='ij')
        rivr_slice = rivr_grid[im, :, :].T  # shape (n_a, n_j)
        
        # Heatmap
        pcm = ax.pcolormesh(A_mesh, J_mesh, rivr_slice, 
                           shading='auto', cmap='viridis')
        
        # Contour lines
        levels = np.linspace(np.nanmin(rivr_slice), np.nanmax(rivr_slice), 8)
        cs = ax.contour(A_mesh, J_mesh, rivr_slice, levels=levels, 
                       colors='white', alpha=0.3, linewidths=0.5)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f Rg')
        
        # ISCO curve
        isco_curve = r_isco(a_vals)
        ax.plot(a_vals, np.ones_like(a_vals) * j_vals[0], 
               color='red', linestyle='--', linewidth=2, label='ISCO')
        
        plt.colorbar(pcm, ax=ax, label='rivr [Rg]')
        
        ax.set_xlabel('a')
        ax.set_ylabel('j')
        ax.set_title(f'm = {mm:.1f}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'rivr heatmap (ν_target = {nu_target:.2e} Hz, M = {M_val:.2e} Msun)', 
                 fontsize=14)
    plt.tight_layout()
    
    return fig, rivr_grid


# --------------------------------------------------------
# ESEMPIO DI UTILIZZO
# --------------------------------------------------------

if __name__ == "__main__":
    # Setup griglia (esempio ridotto per velocità)
    params = {
        "m": [1, 2, 3],
        "j": [1, 2, 3],
        "a": (-0.99, 0.99, 50),
        "r": (1, 100, 100)
    }
    
    param_vectors, mesh_arrays = create_param_grid(params)
    m_grid, j_grid, a_grid, r_grid = mesh_arrays
    
    # Applica vincolo ISCO
    isco_grid = r_isco(a_grid)
    r_grid = np.maximum(r_grid, isco_grid)
    
    # Calcola frequenze (usa la tua funzione c_frq)
    freq_grid = c_frq(m_grid, j_grid, a_grid, r_grid, M_BH)
    
    # Estrai vettori 1D
    m_vals = param_vectors['m']
    j_vals = param_vectors['j']
    a_vals = param_vectors['a']
    r_vals = param_vectors['r']
    
    print("Creazione plot freq vs r con rivr identificato...")
    fig1 = plot_freq_with_rivr(a_vals, r_vals, freq_grid, m_vals, j_vals)
    plt.savefig('freq_vs_r_with_rivr.png', 
                dpi=150, bbox_inches='tight')
    print("Salvato: freq_vs_r_with_rivr.png")
    
    print("\nCreazione heatmap rivr...")
    fig2, rivr_grid = plot_rivr_heatmap(a_vals, m_vals, j_vals, r_vals, freq_grid)
    plt.savefig('rivr_heatmap.png', 
                dpi=150, bbox_inches='tight')
    print("Salvato: rivr_heatmap.png")
    
    # Statistiche
    print(f"\nrivr range: [{np.nanmin(rivr_grid):.2f}, {np.nanmax(rivr_grid):.2f}] Rg")
    print(f"Soluzioni trovate: {np.sum(np.isfinite(rivr_grid))} / {rivr_grid.size}")
    
    plt.show()