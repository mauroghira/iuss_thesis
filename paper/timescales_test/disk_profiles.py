"""
disk_profiles.py

Prescrizioni di disco condivise da PIF e modes per lo studio dei tempi
scala (t_wave, t_align, t_visc). Tutte le grandezze radiali sono in
unita' di r_g = GM/c^2 (stessa convenzione di setup.py: r = R/Rg,
adimensionale), coerentemente con Motta et al. 2017 (arXiv:1709.02608).

Assunzioni dichiarate :
  - p = zeta = 3/5 ovunque (Sigma a torque nullo, radiazione-dominato,
    Franchini et al. 2016)
  - q = 3/2 per il profilo di velocita' del suono (Frank, King & Raine
    1992, Eq. 5.54, disco a pressione di radiazione)
  - (H/R)_ISCO = 0.1 fissato a mano, rappresentativo di AGN vicine a
    Eddington (lambda_Edd ~ 0.1-0.3, vedi RE J1034+396); NON derivato
    da una relazione (H/R)-Mdot esplicita, che le fonti non forniscono
    in forma chiusa utilizzabile qui.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from setup import r_isco, Rg_SUN, C

P_TORQUE = 3.0 / 5.0     # indice del profilo di densita' (= zeta)
Q_SOUND = 3.0 / 2.0      # indice del profilo di velocita' del suono
HR_ISCO_DEFAULT = 0.1    # aspect ratio all'ISCO


def Sigma_torque_zero(R, R_in, p=P_TORQUE):
    """
    Sigma(R) ~ R^-p * (1 - sqrt(R_in/R))^p   (Motta et al. 2017, Eq. 3)

    Normalizzazione arbitraria (Sigma0=1): sia nu_solid sia t_align
    sono indipendenti dalla costante di normalizzazione assoluta. 
    Richiede R >= R_in.
    """
    R = np.asarray(R, dtype=float)
    ratio = np.clip(R_in / R, 0.0, 1.0)
    return R**(-p) * (1.0 - np.sqrt(ratio))**p


def sound_speed_HR0(a, HR_isco=HR_ISCO_DEFAULT, q=Q_SOUND):
    """
    (H/R)_0 = (H/R)_ISCO * r_ISCO^(q - 1/2)   (Motta et al. 2017, testo
    dopo Eq. 6)
    """
    r_isco_val = r_isco(a)
    return HR_isco * r_isco_val**(q - 0.5)


def c_s_powerlaw(r, a, HR_isco=HR_ISCO_DEFAULT, q=Q_SOUND):
    """
    c_s(r) = c * (H/R)_0 * r^-q   [cm/s]

    Non dipende da M: prescrizione puramente geometrica (frazione della
    velocita' della luce), consistente con l'uso in Motta et al. 2017.
    """
    HR0 = sound_speed_HR0(a, HR_isco, q)
    r = np.asarray(r, dtype=float)
    return C * HR0 * r**(-q)


def t_wave_closed(r_in, r_out, a, M, HR_isco=HR_ISCO_DEFAULT, q=Q_SOUND):
    """
    t_wave = 2*(GM/c^3) * (H/R)_ISCO^-1 * r_ISCO^(1/2-q) * (1/(1+q))
             * (r_out^(1+q) - r_in^(1+q))     (Motta et al. 2017, Eq. 6)

    Derivazione: t_wave = int_{r_in}^{r_out} 2 dR/c_s(R), con
    c_s(R) = c_s0 R^-q (adimensionale in unita' di r_g). Risultato in
    secondi (M in masse solari). Vettorizzata su r_in, r_out, a, M.
    """
    r_in = np.asarray(r_in, dtype=float)
    r_out = np.asarray(r_out, dtype=float)
    a = np.asarray(a, dtype=float)
    M = np.asarray(M, dtype=float)

    r_isco_val = r_isco(a)
    Rg_over_c = Rg_SUN * M / C  # GM/c^3, in secondi

    prefactor = 2.0 * Rg_over_c / HR_isco * r_isco_val**(0.5 - q)
    term = (r_out**(1.0 + q) - r_in**(1.0 + q)) / (1.0 + q)
    return prefactor * term
