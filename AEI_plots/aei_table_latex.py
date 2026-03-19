"""
aei_table_latex.py
==================
Produces a LaTeX table summarising, for AEI solutions passing all checks
(WKB + beta + shear), the following statistics over the full
(a, B00, Sigma0) parameter grid:

  - r_min      : minimum physical radius [r_g] of any valid solution
  - B00_min    : minimum B00 [G] for which at least one valid solution exists
  - Sigma0_med : median Sigma0 [g/cm^2] over all valid solutions

Rows: one per (model, H/r) combination
  Models : Simple-v1 (alpha_B=5/4), Simple-v2 (alpha_B=1.7)
  H/r    : 0.1, 0.05, 0.01, 0.001

Usage:
    python aei_table_latex.py
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')

from setup import M_BH, NU0, r_isco
from AEI_setups.aei_common import (
    solve_k_aei, compute_beta, compute_dQdr,
    mm, _make_interp,
)
from AEI_setups.simple_disc import disk_model_simple

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    'Simple-v1': (5/4, 3/5),
    'Simple-v2': (1.7,  3/5),
}

HOR_LIST = [0.1, 0.05, 0.01, 0.001]

A_GRID      = np.linspace(-0.99, 0.99, 49)   # same 19 points as notebook
N_B00       = 24
N_SIGMA0    = 14
B00_GRID    = np.logspace(1, 8, N_B00)
SIGMA0_GRID = np.logspace(2, 7, N_SIGMA0)

N_R        = 300
R_MAX_PHYS = 1000.0
K_MIN_WKB  = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# SCAN — collect all valid (r, B00, Sigma0) triples
# ══════════════════════════════════════════════════════════════════════════════

def collect_valid(alpha_B, alpha_S, hor):
    """
    Return arrays of r [r_g], B00 [G], Sigma0 [g/cm^2] for all valid points.
    """
    k_max_wkb = 1.0 / hor

    r_list, B00_list, Sigma0_list = [], [], []

    total = len(A_GRID) * N_B00 * N_SIGMA0
    count = 0

    # shared r/r_ISCO grid clipped to R_MAX_PHYS per spin
    R_RISCO_GRID = np.geomspace(1.0, R_MAX_PHYS, N_R)  # starts at ISCO

    for a_val in A_GRID:
        isco_val         = float(r_isco(a_val))
        r_risco_max_spin = R_MAX_PHYS / isco_val
        valid_rr         = R_RISCO_GRID[R_RISCO_GRID <= r_risco_max_spin]
        r_vec            = valid_rr * isco_val

        for B00 in B00_GRID:
            for Sigma0 in SIGMA0_GRID:
                count += 1
                if count % 3000 == 0:
                    print(f"  {count}/{total}  ({100*count/total:.0f}%)", end='\r')

                result = disk_model_simple(r_vec, a_val, B00, Sigma0,
                                           alpha_B=alpha_B, alpha_S=alpha_S,
                                           hr=hor, M=M_BH)
                B0_arr, Sigma_arr, cs_arr, hr_arr, _, _ = result

                k_arr = solve_k_aei(r_vec, a_val, B0_arr, Sigma_arr, cs_arr,
                                    m=mm, M=M_BH)

                mask = np.isfinite(k_arr) & (k_arr > 0)
                if not np.any(mask): continue

                mask &= (k_arr >= K_MIN_WKB) & (k_arr <= k_max_wkb)
                if not np.any(mask): continue

                beta_arr = compute_beta(B0_arr, Sigma_arr, cs_arr,
                                        r_vec, hor, M_BH)
                mask &= (beta_arr <= 1.0)
                if not np.any(mask): continue

                _B0i  = _make_interp(r_vec, B0_arr)
                _Sigi = _make_interp(r_vec, Sigma_arr)
                dQdr  = compute_dQdr(r_vec, a_val, _B0i, _Sigi, M_BH)
                mask &= (dQdr > 0)
                if not np.any(mask): continue

                n_valid = np.sum(mask)
                r_list.append(r_vec[mask])
                B00_list.append(np.full(n_valid, B00))
                Sigma0_list.append(np.full(n_valid, Sigma0))

    print()

    if not r_list:
        return np.array([]), np.array([]), np.array([])
    return (np.concatenate(r_list),
            np.concatenate(B00_list),
            np.concatenate(Sigma0_list))

# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLE
# ══════════════════════════════════════════════════════════════════════════════

def sci(x, decimals=1):
    """Format a number in LaTeX scientific notation."""
    if not np.isfinite(x):
        return r'---'
    exp = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exp
    if decimals == 0:
        return rf'$10^{{{exp}}}$'
    return rf'${coeff:.{decimals}f} \times 10^{{{exp}}}$'

def build_latex_table(rows):
    """
    rows : list of dicts with keys:
        model, hor, r_min, B00_min, Sigma0_med
    """
    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{Statistics of AEI-valid solutions (all checks:'
                 r' WKB + $\beta$ + shear) for Simple-v1 ($\alpha_B=5/4$)'
                 r' and Simple-v2 ($\alpha_B=1.7$), both with $\alpha_\Sigma=3/5$.'
                 r' $r_{\min}$: minimum radius of any valid solution [r$_g$];'
                 r' $B_{00,\min}$: minimum $B_{00}$ with at least one valid solution [G];'
                 r' $\Sigma_{0,\rm med}$: median $\Sigma_0$ over all valid solutions [g\,cm$^{-2}$].}')
    lines.append(r'\label{tab:aei_stats}')
    lines.append(r'\begin{tabular}{llcccc}')
    lines.append(r'\hline\hline')
    lines.append(r'Model & $H/r$ & $r_{\min}$ [r$_g$] & '
                 r'$B_{00}^{\min}$ [G] & $\langle \Sigma_0 \rangle$ [g\,cm$^{-2}$] \\')
    lines.append(r'\hline')

    prev_model = None
    for row in rows:
        model_str = row['model'] if row['model'] != prev_model else ''
        prev_model = row['model']
        hor_str = f"${row['hor']}$"
        lines.append(
            rf"{model_str} & {hor_str} & "
            rf"{sci(row['r_min'])} & "
            rf"{sci(row['B00_min'])} & "
            rf"{sci(row['Sigma0_med'])} \\"
        )
        if model_str != '' and model_str == list(MODELS.keys())[-1]:
            pass
        elif model_str == '' and row['hor'] == HOR_LIST[-1]:
            lines.append(r'\hline')

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  AEI LaTeX table — Simple-v1 & v2, 4 H/r values")
    print(f"  Spin  : {len(A_GRID)} values  |  B00: {N_B00}  |  Sigma0: {N_SIGMA0}")
    print("=" * 65)

    rows = []

    for model_label, (alpha_B, alpha_S) in MODELS.items():
        for hor in HOR_LIST:
            print(f"\n> {model_label}  H/r={hor}  (k_max={1/hor:.0f})")
            r_arr, B00_arr, S0_arr = collect_valid(alpha_B, alpha_S, hor)

            if r_arr.size == 0:
                print("   -> no valid solutions found")
                rows.append(dict(model=model_label, hor=hor,
                                 r_min=np.nan, B00_min=np.nan,
                                 Sigma0_med=np.nan))
                continue

            r_min      = r_arr.min()
            B00_min    = B00_arr.min()
            Sigma0_med = np.median(S0_arr)

            print(f"   -> r_min={r_min:.3f} rg  "
                  f"B00_min={B00_min:.2e} G  Sigma0_med={Sigma0_med:.2e}")

            rows.append(dict(model=model_label, hor=hor,
                             r_min=r_min, B00_min=B00_min,
                             Sigma0_med=Sigma0_med))

    print("\n" + "=" * 65)
    print("  LaTeX table:")
    print("=" * 65)
    table = build_latex_table(rows)
    print(table)

    with open('aei_table.tex', 'w') as f:
        f.write(table + '\n')
    print("\n   -> saved aei_table.tex")
