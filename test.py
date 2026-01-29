"""
Accretion-Ejection Instability Analysis for J1257 QPO
Based on Tagger & Pellat (1999) model

This code explores the parameter space of the accretion-ejection instability
to find constraints matching the observed QPO frequency in J1257.

Author: Analysis for J1257 AGN QPO study
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Physical constants
G = 6.67430e-8  # cm^3 g^-1 s^-2
c = 2.99792458e10  # cm/s
M_sun = 1.989e33  # g
sigma_sb = 5.670374419e-5  # erg cm^-2 s^-1 K^-4
k_B = 1.380649e-16  # erg/K
m_p = 1.672621e-24  # g (proton mass)
mu = 0.6  # mean molecular weight

class BlackHole:
    """Black hole properties"""
    def __init__(self, mass_msun, spin=0.0):
        """
        Parameters:
        -----------
        mass_msun : float
            Black hole mass in solar masses
        spin : float
            Dimensionless spin parameter (-1 to 1)
        """
        self.M = mass_msun * M_sun  # grams
        self.M_msun = mass_msun
        self.a = spin  # dimensionless spin
        
        # Gravitational radius
        self.r_g = G * self.M / c**2  # cm
        
        # ISCO radius (Bardeen et al. 1972)
        Z1 = 1 + (1 - self.a**2)**(1/3) * ((1 + self.a)**(1/3) + (1 - self.a)**(1/3))
        Z2 = np.sqrt(3 * self.a**2 + Z1**2)
        
        if self.a >= 0:
            self.r_isco = self.r_g * (3 + Z2 - np.sqrt((3 - Z1) * (3 + Z1 + 2*Z2)))
        else:
            self.r_isco = self.r_g * (3 + Z2 + np.sqrt((3 - Z1) * (3 + Z1 + 2*Z2)))
    
    def orbital_frequency(self, r):
        """
        Keplerian orbital frequency at radius r
        
        Parameters:
        -----------
        r : float or array
            Radius in cm
            
        Returns:
        --------
        nu : float or array
            Frequency in Hz
        """
        omega = np.sqrt(G * self.M / r**3)
        return omega / (2 * np.pi)
    
    def angular_velocity(self, r):
        """Angular velocity Omega_K"""
        return np.sqrt(G * self.M / r**3)


class AccretionDisk:
    """
    Standard thin disk model (Shakura-Sunyaev 1973)
    with magnetic field prescription
    """
    def __init__(self, bh, mdot_edd=0.1, alpha=0.1):
        """
        Parameters:
        -----------
        bh : BlackHole object
            The central black hole
        mdot_edd : float
            Accretion rate in units of Eddington rate
        alpha : float
            Shakura-Sunyaev viscosity parameter
        """
        self.bh = bh
        self.alpha = alpha
        
        # Eddington accretion rate
        eta = 0.1  # radiative efficiency
        L_edd = 1.26e38 * bh.M_msun  # erg/s
        self.mdot_edd = L_edd / (eta * c**2)  # g/s
        
        self.mdot = mdot_edd * self.mdot_edd
        self.mdot_edd_fraction = mdot_edd
        
    def temperature(self, r):
        """
        Disk temperature at radius r (Shakura-Sunyaev)
        
        Returns:
        --------
        T : float or array
            Temperature in Kelvin
        """
        r_g = self.bh.r_g
        x = r / r_g
        
        # Inner boundary correction
        x_in = self.bh.r_isco / r_g
        f = 1 - np.sqrt(x_in / x)
        
        T = 1.4e8 * self.mdot_edd_fraction**(1/4) * self.bh.M_msun**(-1/4) * x**(-3/4) * f**(1/4)
        return T
    
    def surface_density(self, r):
        """
        Surface density Sigma at radius r
        
        Returns:
        --------
        Sigma : float or array
            Surface density in g/cm^2
        """
        r_g = self.bh.r_g
        x = r / r_g
        x_in = self.bh.r_isco / r_g
        
        f = 1 - np.sqrt(x_in / x)
        
        Sigma = 1.4e5 * self.alpha**(-4/5) * self.mdot_edd_fraction**(3/5) * \
                self.bh.M_msun**(1/5) * x**(-3/5) * f**(4/5)
        return Sigma
    
    def scale_height(self, r):
        """
        Disk scale height H at radius r
        
        Returns:
        --------
        H : float or array
            Scale height in cm
        """
        T = self.temperature(r)
        Omega = self.bh.angular_velocity(r)
        
        c_s = np.sqrt(k_B * T / (mu * m_p))
        H = c_s / Omega
        return H
    
    def midplane_density(self, r):
        """
        Midplane volume density at radius r
        
        Returns:
        --------
        rho : float or array
            Density in g/cm^3
        """
        Sigma = self.surface_density(r)
        H = self.scale_height(r)
        
        # Gaussian vertical structure
        rho = Sigma / (np.sqrt(2 * np.pi) * H)
        return rho
    
    def sound_speed(self, r):
        """
        Sound speed at radius r
        
        Returns:
        --------
        c_s : float or array
            Sound speed in cm/s
        """
        T = self.temperature(r)
        c_s = np.sqrt(k_B * T / (mu * m_p))
        return c_s
    
    def magnetic_field(self, r, B0=None, r0=None, beta=5/4):
        """
        Magnetic field at radius r
        
        Parameters:
        -----------
        r : float or array
            Radius in cm
        B0 : float
            Magnetic field strength at r0 (Gauss)
        r0 : float
            Reference radius (default: ISCO)
        beta : float
            Power law index (default: 5/4 for equipartition)
            
        Returns:
        --------
        B : float or array
            Magnetic field in Gauss
        """
        if r0 is None:
            r0 = self.bh.r_isco
            
        if B0 is None:
            # Estimate from equipartition with thermal pressure
            rho0 = self.midplane_density(r0)
            c_s0 = self.sound_speed(r0)
            P_thermal = rho0 * c_s0**2
            
            # Assume beta_plasma ~ 1-10 (magnetic pressure ~ thermal pressure)
            beta_plasma = 5.0
            B0 = np.sqrt(8 * np.pi * P_thermal / beta_plasma)
        
        B = B0 * (r / r0)**(-beta)
        return B


class TaggerPellatInstability:
    """
    Accretion-Ejection Instability (Tagger & Pellat 1999)
    
    The instability arises from the coupling between the disk and
    a large-scale magnetic field, leading to quasi-periodic oscillations.
    """
    
    def __init__(self, disk):
        """
        Parameters:
        -----------
        disk : AccretionDisk object
            The accretion disk model
        """
        self.disk = disk
        self.bh = disk.bh
        
    def alfven_velocity(self, r, B):
        """
        Alfvén velocity
        
        Parameters:
        -----------
        r : float
            Radius in cm
        B : float
            Magnetic field in Gauss
            
        Returns:
        --------
        v_A : float
            Alfvén velocity in cm/s
        """
        rho = self.disk.midplane_density(r)
        v_A = B / np.sqrt(4 * np.pi * rho)
        return v_A
    
    def mach_number_alfven(self, r, B):
        """
        Alfvénic Mach number M_A = c_s / v_A
        """
        c_s = self.disk.sound_speed(r)
        v_A = self.alfven_velocity(r, B)
        return c_s / v_A
    
    def dispersion_relation_simplified(self, omega, r, B, m=1):
        """
        Simplified dispersion relation for the instability
        (Following Tagger & Pellat 1999, eq. 14-16)
        
        For the Rossby wave instability coupled to magnetic field:
        omega^2 = Omega_K^2 * [1 - (m * v_A / (r * Omega_K))^2]
        
        The fastest growing mode occurs when the Alfvén crossing time
        matches the orbital time.
        
        Parameters:
        -----------
        omega : complex
            Angular frequency
        r : float
            Radius
        B : float
            Magnetic field
        m : int
            Azimuthal mode number
            
        Returns:
        --------
        residual : complex
            Dispersion relation residual
        """
        Omega_K = self.bh.angular_velocity(r)
        v_A = self.alfven_velocity(r, B)
        c_s = self.disk.sound_speed(r)
        
        # Characteristic frequency (from Tagger & Pellat)
        # The instability frequency is related to:
        # 1) Orbital frequency Omega_K
        # 2) Alfvén wave crossing time
        # 3) Rossby wave frequency
        
        k_r = m / r  # radial wavenumber (simplified)
        
        # Dispersion relation (simplified form)
        omega_R = Omega_K * (1 - m * v_A / (r * Omega_K))  # Rossby-like
        omega_A = k_r * v_A  # Alfvén
        
        # The instability occurs when these frequencies match
        # Growth rate is maximum when omega_A ~ Omega_K
        
        return omega**2 - omega_R * omega_A
    
    def characteristic_frequency(self, r, B, m=1):
        """
        Characteristic frequency of the instability
        
        Following Tagger & Pellat (1999), the frequency is roughly:
        nu ~ (v_A / r) * (Omega_K * r / v_A)^(1/2)
        
        Or more simply, when Alfvén crossing time ~ orbital period:
        nu ~ Omega_K * M_A^(-1)
        
        where M_A is the Alfvénic Mach number
        
        Returns:
        --------
        nu : float
            Frequency in Hz
        """
        Omega_K = self.bh.angular_velocity(r)
        v_A = self.alfven_velocity(r, B)
        c_s = self.disk.sound_speed(r)
        
        # Method 1: Alfvén-rotation coupling
        # The instability frequency scales as the geometric mean
        nu_1 = (Omega_K * v_A / r) / (2 * np.pi)
        
        # Method 2: Modified by sound speed
        M_A = c_s / v_A
        nu_2 = (Omega_K / (2 * np.pi)) * M_A
        
        # Method 3: Rossby wave frequency (Tagger & Pellat 1999)
        # For m=1 mode
        nu_3 = (Omega_K / (2 * np.pi)) * (1 - v_A / (r * Omega_K / m))
        
        # We use Method 1 as the primary estimate
        # (this matches the physics of magnetic/rotation coupling)
        return nu_1
    
    def find_resonance_radius(self, nu_obs, B0, r0=None, r_range=None):
        """
        Find the radius where the instability frequency matches observation
        
        Parameters:
        -----------
        nu_obs : float
            Observed frequency in Hz
        B0 : float
            Magnetic field at r0 in Gauss
        r0 : float
            Reference radius (default: 6 r_g)
        r_range : tuple
            (r_min, r_max) search range in r_g units
            
        Returns:
        --------
        r_res : float
            Resonance radius in cm (or None if not found)
        """
        if r0 is None:
            r0 = 6 * self.bh.r_g
        
        if r_range is None:
            r_range = (3, 500)  # in r_g units (extended range for low frequencies)
        
        r_min = r_range[0] * self.bh.r_g
        r_max = r_range[1] * self.bh.r_g
        
        def residual(r_log):
            r = 10**r_log
            if r < r_min or r > r_max:
                return 1e10
            
            B = self.disk.magnetic_field(r, B0=B0, r0=r0)
            nu_model = self.characteristic_frequency(r, B)
            
            return (nu_model - nu_obs)**2
        
        # Search in log space
        r_log_min = np.log10(r_min)
        r_log_max = np.log10(r_max)
        r_log_guess = (r_log_min + r_log_max) / 2
        
        result = minimize(residual, r_log_guess, 
                         bounds=[(r_log_min, r_log_max)],
                         method='L-BFGS-B')
        
        if result.success and result.fun < (nu_obs * 0.01)**2:
            return 10**result.x[0]
        else:
            return None


def analyze_j1257():
    """
    Analysis specific to J1257 AGN
    """
    # Observed parameters from the paper
    nu_obs = 3.3e-5  # Hz
    M_BH = 10**(6.3)  # Solar masses
    z = 0.02068
    
    print("="*70)
    print("TAGGER-PELLAT INSTABILITY ANALYSIS FOR J1257")
    print("="*70)
    print(f"\nObserved QPO frequency: {nu_obs*1e6:.2f} μHz")
    print(f"Black hole mass: {M_BH:.2e} M_sun")
    print(f"Redshift: {z}")
    print("\n" + "="*70)
    
    # Create parameter grid
    spins = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 0.998])
    mdot_edds = np.array([0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5])
    B0_values = np.logspace(0, 6, 100)  # Gauss (1 - 1,000,000 G)
    
    results = []
    
    for spin in spins:
        bh = BlackHole(M_BH, spin=spin)
        
        print(f"\n--- Spin a* = {spin} ---")
        print(f"ISCO radius: {bh.r_isco/bh.r_g:.2f} r_g = {bh.r_isco/1e13:.2e} × 10^13 cm")
        print(f"Orbital frequency at ISCO: {bh.orbital_frequency(bh.r_isco)*1e6:.2f} μHz")
        
        for mdot_edd in mdot_edds:
            disk = AccretionDisk(bh, mdot_edd=mdot_edd, alpha=0.1)
            instability = TaggerPellatInstability(disk)
            
            # Reference radius (typically a few r_g)
            r0 = 6 * bh.r_g
            
            for B0 in B0_values:
                r_res = instability.find_resonance_radius(nu_obs, B0, r0=r0)
                
                if r_res is not None:
                    # Calculate disk properties at resonance
                    B_res = disk.magnetic_field(r_res, B0=B0, r0=r0)
                    T_res = disk.temperature(r_res)
                    rho_res = disk.midplane_density(r_res)
                    c_s_res = disk.sound_speed(r_res)
                    v_A_res = instability.alfven_velocity(r_res, B_res)
                    M_A_res = c_s_res / v_A_res
                    
                    results.append({
                        'spin': spin,
                        'mdot_edd': mdot_edd,
                        'B0': B0,
                        'r_res_rg': r_res / bh.r_g,
                        'r_res_cm': r_res,
                        'B_res': B_res,
                        'T_res': T_res,
                        'rho_res': rho_res,
                        'c_s_res': c_s_res,
                        'v_A_res': v_A_res,
                        'M_A': M_A_res
                    })
    
    print(f"\n\nFound {len(results)} parameter combinations matching the observed frequency!")
    
    return results, bh, disk, instability


def plot_results(results, nu_obs):
    """
    Create comprehensive plots of the results
    """
    if len(results) == 0:
        print("No matching solutions found!")
        return
    
    # Convert to arrays
    spins = np.array([r['spin'] for r in results])
    mdots = np.array([r['mdot_edd'] for r in results])
    B0s = np.array([r['B0'] for r in results])
    r_res = np.array([r['r_res_rg'] for r in results])
    B_res = np.array([r['B_res'] for r in results])
    M_As = np.array([r['M_A'] for r in results])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Tagger-Pellat Instability: Parameter Space for ν = {nu_obs*1e6:.2f} μHz', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: B0 vs radius (colored by spin)
    scatter1 = axes[0, 0].scatter(r_res, B0s, c=spins, cmap='viridis', 
                                   s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0, 0].set_xlabel('Resonance Radius [r_g]', fontsize=12)
    axes[0, 0].set_ylabel('B₀ at 6 r_g [G]', fontsize=12)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title('Magnetic Field vs Radius')
    axes[0, 0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Spin a*', fontsize=10)
    
    # Plot 2: B0 vs M_A (colored by mdot)
    scatter2 = axes[0, 1].scatter(M_As, B0s, c=mdots, cmap='plasma',
                                   s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0, 1].set_xlabel('Alfvénic Mach Number M_A', fontsize=12)
    axes[0, 1].set_ylabel('B₀ at 6 r_g [G]', fontsize=12)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_title('Magnetic Field vs Mach Number')
    axes[0, 1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('ṁ/ṁ_Edd', fontsize=10)
    
    # Plot 3: Radius vs Spin (colored by B0)
    scatter3 = axes[0, 2].scatter(spins, r_res, c=np.log10(B0s), cmap='coolwarm',
                                   s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    axes[0, 2].set_xlabel('Spin a*', fontsize=12)
    axes[0, 2].set_ylabel('Resonance Radius [r_g]', fontsize=12)
    axes[0, 2].set_title('Radius vs Spin')
    axes[0, 2].grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=axes[0, 2])
    cbar3.set_label('log₁₀(B₀ [G])', fontsize=10)
    
    # Plot 4: Distribution of resonance radii
    axes[1, 0].hist(r_res, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1, 0].axvline(np.median(r_res), color='red', linestyle='--', 
                      linewidth=2, label=f'Median: {np.median(r_res):.1f} r_g')
    axes[1, 0].set_xlabel('Resonance Radius [r_g]', fontsize=12)
    axes[1, 0].set_ylabel('Number of Solutions', fontsize=12)
    axes[1, 0].set_title('Distribution of Resonance Radii')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Distribution of B field at resonance
    axes[1, 1].hist(B_res, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[1, 1].axvline(np.median(B_res), color='darkred', linestyle='--',
                      linewidth=2, label=f'Median: {np.median(B_res):.0f} G')
    axes[1, 1].set_xlabel('B field at resonance [G]', fontsize=12)
    axes[1, 1].set_ylabel('Number of Solutions', fontsize=12)
    axes[1, 1].set_title('Distribution of Magnetic Field Strength')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Parameter correlation matrix
    unique_spins = np.unique(spins)
    unique_mdots = np.unique(mdots)
    
    # Create 2D histogram of solutions in spin-mdot space
    H, xedges, yedges = np.histogram2d(spins, mdots, 
                                       bins=[len(unique_spins), len(unique_mdots)])
    
    im = axes[1, 2].imshow(H.T, origin='lower', aspect='auto', cmap='YlOrRd',
                          extent=[unique_spins.min(), unique_spins.max(),
                                  unique_mdots.min(), unique_mdots.max()])
    axes[1, 2].set_xlabel('Spin a*', fontsize=12)
    axes[1, 2].set_ylabel('ṁ/ṁ_Edd', fontsize=12)
    axes[1, 2].set_title('Solution Density in Parameter Space')
    cbar6 = plt.colorbar(im, ax=axes[1, 2])
    cbar6.set_label('N solutions', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def print_summary_table(results, n_samples=10):
    """
    Print a summary table of representative solutions
    """
    if len(results) == 0:
        return
    
    print("\n" + "="*100)
    print("REPRESENTATIVE SOLUTIONS (sample of {})".format(min(n_samples, len(results))))
    print("="*100)
    print(f"{'Spin':>6} {'ṁ/ṁ_Edd':>9} {'r_res':>8} {'B₀':>10} {'B_res':>10} {'M_A':>8} {'T':>10} {'ρ':>12}")
    print(f"{'a*':>6} {'':>9} {'[r_g]':>8} {'[G]':>10} {'[G]':>10} {'':>8} {'[K]':>10} {'[g/cm³]':>12}")
    print("-"*100)
    
    # Sample evenly through the results
    indices = np.linspace(0, len(results)-1, min(n_samples, len(results)), dtype=int)
    
    for i in indices:
        r = results[i]
        print(f"{r['spin']:6.2f} {r['mdot_edd']:9.2f} {r['r_res_rg']:8.2f} "
              f"{r['B0']:10.1f} {r['B_res']:10.1f} {r['M_A']:8.2f} "
              f"{r['T_res']:10.1e} {r['rho_res']:12.2e}")
    
    print("-"*100)
    
    # Print statistics
    r_res_all = np.array([r['r_res_rg'] for r in results])
    B0_all = np.array([r['B0'] for r in results])
    B_res_all = np.array([r['B_res'] for r in results])
    M_A_all = np.array([r['M_A'] for r in results])
    
    print(f"\nSTATISTICS OVER ALL {len(results)} SOLUTIONS:")
    print(f"  Resonance radius:     {np.min(r_res_all):6.2f} - {np.max(r_res_all):6.2f} r_g "
          f"(median: {np.median(r_res_all):.2f})")
    print(f"  B₀ at 6 r_g:          {np.min(B0_all):6.1f} - {np.max(B0_all):6.1f} G "
          f"(median: {np.median(B0_all):.1f})")
    print(f"  B at resonance:       {np.min(B_res_all):6.1f} - {np.max(B_res_all):6.1f} G "
          f"(median: {np.median(B_res_all):.1f})")
    print(f"  Alfvénic Mach number: {np.min(M_A_all):6.2f} - {np.max(M_A_all):6.2f} "
          f"(median: {np.median(M_A_all):.2f})")
    print("="*100)


def frequency_vs_radius_plot(bh_mass=10**6.3, spin=0.0, mdot_edd=0.1, B0=1000):
    """
    Create a diagnostic plot showing how frequency varies with radius
    for a given set of parameters
    """
    bh = BlackHole(bh_mass, spin=spin)
    disk = AccretionDisk(bh, mdot_edd=mdot_edd)
    instability = TaggerPellatInstability(disk)
    
    # Radius range
    r_array = np.logspace(np.log10(3*bh.r_g), np.log10(100*bh.r_g), 200)
    r_rg = r_array / bh.r_g
    
    # Reference radius
    r0 = 6 * bh.r_g
    
    # Calculate frequencies and other quantities
    frequencies = []
    B_values = []
    M_A_values = []
    
    for r in r_array:
        B = disk.magnetic_field(r, B0=B0, r0=r0)
        nu = instability.characteristic_frequency(r, B)
        v_A = instability.alfven_velocity(r, B)
        c_s = disk.sound_speed(r)
        M_A = c_s / v_A
        
        frequencies.append(nu)
        B_values.append(B)
        M_A_values.append(M_A)
    
    frequencies = np.array(frequencies)
    B_values = np.array(B_values)
    M_A_values = np.array(M_A_values)
    
    # Observed frequency
    nu_obs = 3.3e-5
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Frequency Dependence on Radius\n' + 
                 f'M = {bh_mass:.2e} M☉, a* = {spin}, ṁ/ṁ_Edd = {mdot_edd}, B₀ = {B0} G',
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Frequency vs radius
    axes[0, 0].loglog(r_rg, frequencies * 1e6, 'b-', linewidth=2, label='Model')
    axes[0, 0].axhline(nu_obs * 1e6, color='red', linestyle='--', linewidth=2, 
                      label=f'Observed: {nu_obs*1e6:.2f} μHz')
    axes[0, 0].set_xlabel('Radius [r_g]', fontsize=12)
    axes[0, 0].set_ylabel('Frequency [μHz]', fontsize=12)
    axes[0, 0].set_title('Instability Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Magnetic field vs radius
    axes[0, 1].loglog(r_rg, B_values, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Radius [r_g]', fontsize=12)
    axes[0, 1].set_ylabel('Magnetic Field [G]', fontsize=12)
    axes[0, 1].set_title('B(r) Profile')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Alfvénic Mach number vs radius
    axes[1, 0].semilogx(r_rg, M_A_values, 'orange', linewidth=2)
    axes[1, 0].set_xlabel('Radius [r_g]', fontsize=12)
    axes[1, 0].set_ylabel('M_A = c_s / v_A', fontsize=12)
    axes[1, 0].set_title('Alfvénic Mach Number')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Temperature and density
    ax4a = axes[1, 1]
    T_values = [disk.temperature(r) for r in r_array]
    ax4a.loglog(r_rg, T_values, 'r-', linewidth=2, label='Temperature')
    ax4a.set_xlabel('Radius [r_g]', fontsize=12)
    ax4a.set_ylabel('Temperature [K]', fontsize=12, color='r')
    ax4a.tick_params(axis='y', labelcolor='r')
    ax4a.grid(True, alpha=0.3)
    
    ax4b = ax4a.twinx()
    rho_values = [disk.midplane_density(r) for r in r_array]
    ax4b.loglog(r_rg, rho_values, 'b-', linewidth=2, label='Density')
    ax4b.set_ylabel('Density [g/cm³]', fontsize=12, color='b')
    ax4b.tick_params(axis='y', labelcolor='b')
    
    axes[1, 1].set_title('Disk Structure')
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Run the main analysis
    results, bh, disk, instability = analyze_j1257()
    
    # Print summary table
    print_summary_table(results, n_samples=20)
    
    # Create plots
    if len(results) > 0:
        fig1 = plot_results(results, 3.3e-5)
        fig1.savefig('/home/claude/j1257_parameter_space.png', dpi=150, bbox_inches='tight')
        print("\nParameter space plot saved as 'j1257_parameter_space.png'")
        
        # Create diagnostic plot for a representative case
        fig2 = frequency_vs_radius_plot(bh_mass=10**6.3, spin=0.5, mdot_edd=0.1, B0=500)
        fig2.savefig('/home/claude/j1257_frequency_diagnostic.png', dpi=150, bbox_inches='tight')
        print("Frequency diagnostic plot saved as 'j1257_frequency_diagnostic.png'")
        
        plt.show()
    else:
        print("\nNo solutions found. Try adjusting the parameter ranges.")
        
        # Still create diagnostic plot
        fig2 = frequency_vs_radius_plot(bh_mass=10**6.3, spin=0.5, mdot_edd=0.1, B0=500)
        fig2.savefig('/home/claude/j1257_frequency_diagnostic.png', dpi=150, bbox_inches='tight')
        print("Frequency diagnostic plot saved as 'j1257_frequency_diagnostic.png'")
        plt.show()