"""Shared disk prescriptions for the PIF timescale calculations."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from setup import C, Rg_SUN, r_isco

P_TORQUE = 3.0 / 5.0
Q_SOUND = 3.0 / 2.0
HR_ISCO_DEFAULT = 0.1


def Sigma_torque_zero(R, R_in, p=P_TORQUE):
    """Torque-free surface-density profile, normalized to Sigma0=1."""
    R = np.asarray(R, dtype=float)
    ratio = np.clip(R_in / R, 0.0, 1.0)
    return R**(-p) * (1.0 - np.sqrt(ratio))**p


def sound_speed_HR0(a, HR_isco=HR_ISCO_DEFAULT, q=Q_SOUND):
    """Aspect-ratio normalization at the ISCO."""
    return HR_isco * r_isco(a)**(q - 0.5)


def c_s_powerlaw(r, a, HR_isco=HR_ISCO_DEFAULT, q=Q_SOUND):
    """Sound speed profile in cm/s."""
    HR0 = sound_speed_HR0(a, HR_isco, q)
    return C * HR0 * np.asarray(r, dtype=float)**(-q)


def t_wave_closed(r_in, r_out, a, M, HR_isco=HR_ISCO_DEFAULT, q=Q_SOUND):
    """Closed-form wave-propagation time in seconds."""
    r_in = np.asarray(r_in, dtype=float)
    r_out = np.asarray(r_out, dtype=float)
    a = np.asarray(a, dtype=float)
    M = np.asarray(M, dtype=float)

    r_isco_val = r_isco(a)
    Rg_over_c = Rg_SUN * M / C
    prefactor = 2.0 * Rg_over_c / HR_isco * r_isco_val**(0.5 - q)
    term = (r_out**(1.0 + q) - r_in**(1.0 + q)) / (1.0 + q)
    return prefactor * term
