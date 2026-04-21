"""
Disk Oscillation Modes — Publication Figure
============================================
Griglia 3×3: righe = modi (p, g, c), colonne = snapshot temporali.
Output: disk_modes_snapshots.pdf  +  disk_modes_snapshots.png (300 dpi)

Dipendenze: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.colors import Normalize, LightSource
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
from setup import set_style, fix_spines

set_style()

# ──────────────────────────────────────────────────────────────
# PARAMETRI
# ──────────────────────────────────────────────────────────────
R_IN        = 3.0
R_OUT       = 12.0
H_OVER_R    = 0.12
R_TRAP_IN   = 4.2
R_TRAP_OUT  = 9.5
Nr, Nphi    = 70, 110

r   = np.linspace(R_IN, R_OUT, Nr)
phi = np.linspace(0, 2 * np.pi, Nphi)
R, PHI = np.meshgrid(r, phi)

X0 = R * np.cos(PHI)
Y0 = R * np.sin(PHI)
H_loc = H_OVER_R * R

# Inviluppo di trapping
r0    = 0.5 * (R_TRAP_IN + R_TRAP_OUT)
sigma = (R_TRAP_OUT - R_TRAP_IN) / 4.5

def envelope(R):
    e = np.exp(-((R - r0)**2) / (2 * sigma**2))
    e = np.where(R < R_TRAP_IN,
                 e * np.exp(-((R - R_TRAP_IN)**2) / 0.4**2), e)
    e = np.where(R > R_TRAP_OUT,
                 e * np.exp(-((R - R_TRAP_OUT)**2) / 0.4**2), e)
    return e

ENV = envelope(R)
k_p = 2.0 * np.pi / (R_TRAP_OUT - R_TRAP_IN)
k_g = 1.5 * np.pi / (R_TRAP_OUT - R_TRAP_IN)
k_c = 0.8 * np.pi / (R_TRAP_OUT - R_TRAP_IN)

# ──────────────────────────────────────────────────────────────
# FUNZIONI DEI MODI
# ──────────────────────────────────────────────────────────────

def p_mode(t, A=0.9):
    """n=0, m=0: compressione radiale, piano piatto."""
    spatial = ENV * np.cos(k_p * R)          # struttura spaziale fissa
    dr    = 0.18 * A * spatial * np.cos(t)   # onda stazionaria
    drho  = A * spatial * np.cos(t)
    X     = (R + dr) * np.cos(PHI)
    Y     = (R + dr) * np.sin(PHI)
    Z     = np.zeros_like(R)
    return X, Y, Z, drho

def g_mode(t, A=1.0):
    """n=1, m=0: oscillazione verticale antisimmetrica (onda stazionaria).
    amp = ENV * cos(k_g*R) * cos(t)  →  zero globale esatto a t=pi/2.
    """
    radial   = np.cos(k_g * R)          # struttura spaziale fissa
    amp      = A * H_loc * ENV * radial
    Z_upper  =  amp
    Z_lower  = -amp
    drho     = A * ENV * radial * (np.cos(t))**2
    return X0, Y0, Z_upper, Z_lower, drho

def c_mode(t, A=1.0):
    """n=1, m=1: corrugazione del piano, precessione lenta."""
    Z    = A * H_loc * ENV * np.cos(PHI - t - np.pi/2) * np.cos(k_c * (R - r0))
    drho = 0.35 * A * ENV * np.cos(PHI - t - np.pi/2) * np.cos(k_c * (R - r0))
    return X0, Y0, Z, drho

# ──────────────────────────────────────────────────────────────
# TEMPI DEGLI SNAPSHOT
# ──────────────────────────────────────────────────────────────
T_SNAP   = [0.0, np.pi / 4, np.pi / 2]
T_LABELS = [r"$\omega t = 0$",
            r"$\omega t = \pi/4$",
            r"$\omega t = \pi/2$"]

# ──────────────────────────────────────────────────────────────
# STILE PUBBLICAZIONE
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':         8,
    'axes.labelsize':    8,
    'axes.titlesize':    9,
    'text.usetex':       False,
    'figure.dpi':        150,
})

CMAP      = cm.RdBu_r
NORM      = Normalize(vmin=-1, vmax=1)
BG        = 'white'
GRID_COL  = '#cccccc'
PANE_COL  = 'white'

# colori etichette riga
ROW_COLORS = ['#1a6bbf', '#b07800', '#1a8a1a']
ROW_TITLES = [
    r"p-mode  ($n{=}0,\,m{=}0$)",
    r"g-mode  ($n{=}1,\,m{=}0$)",
    r"c-mode  ($n{=}1,\,m{=}1$)",
]

ELEV, AZIM = 24, -52

# ──────────────────────────────────────────────────────────────
# FIGURA
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 5.5), facecolor='white')

# Riserva colonna sinistra per le etichette di riga e colonna destra per colorbar
gs_outer = gridspec.GridSpec(
    1, 3,
    figure=fig,
    left=0.015, right=0.87,
    top=0.91, bottom=0.04,
    wspace=0.02
)

# Per ogni colonna (snapshot), 3 righe (modi)
axes = []   # axes[row][col]
for col in range(3):
    gs_inner = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=gs_outer[col],
        hspace=0.08
    )
    col_axes = []
    for row in range(3):
        ax = fig.add_subplot(gs_inner[row], projection='3d')
        col_axes.append(ax)
    axes.append(col_axes)

# axes[col][row]  →  vogliamo axes[row][col], trasponiamo
axes = [[axes[col][row] for col in range(3)] for row in range(3)]

# ──────────────────────────────────────────────────────────────
# FUNZIONE DI STILE ASSE
# ──────────────────────────────────────────────────────────────
ls = LightSource(azdeg=225, altdeg=35)

def style_ax(ax, zlim=1.6):
    ax.set_facecolor(PANE_COL)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.35, alpha=0.7)
    ax.tick_params(colors='#555555', labelsize=5, pad=-3,
                   length=2, width=0.5)
    ax.set_xlim(-R_OUT, R_OUT)
    ax.set_ylim(-R_OUT, R_OUT)
    ax.set_zlim(-zlim, zlim)
    ax.set_xticks([-8, 0, 8])
    ax.set_yticks([-8, 0, 8])
    ax.set_zticks([-1, 0, 1])
    ax.set_xlabel(r'$x\,[r_g]$', color='#333333',
                  labelpad=-7)
    ax.set_ylabel(r'$y\,[r_g]$', color='#333333',
                  labelpad=-7)
    ax.set_zlabel(r'$z/H$', fontsize=7, color='#333333',labelpad=-9)
        
    ax.view_init(elev=ELEV, azim=AZIM)

def draw_midplane(ax):
    """Disco piatto di riferimento non perturbato."""
    ax.plot_wireframe(
        X0, Y0, np.zeros_like(X0),
        rstride=12, cstride=12,
        color='#aaaaaa', alpha=0.45, linewidth=0.25
    )

# ──────────────────────────────────────────────────────────────
# POPOLAMENTO DELLA GRIGLIA
# ──────────────────────────────────────────────────────────────
for col, t in enumerate(T_SNAP):

    # ── P-MODE (riga 0) ──────────────────────────────────────
    ax = axes[0][col]
    Xp, Yp, Zp, drho_p = p_mode(t)
    fc_p = CMAP(NORM(drho_p))
    draw_midplane(ax)
    ax.plot_surface(Xp, Yp, Zp,
                    facecolors=fc_p,
                    rstride=2, cstride=2,
                    alpha=0.88, shade=True)
    # cerchi iso-fase
    for rr in np.linspace(R_TRAP_IN + 0.5, R_TRAP_OUT - 0.5, 4):
        env_r = np.exp(-((rr - r0)**2) / (2 * sigma**2))
        dr_c  = 0.18 * env_r * np.cos(k_p * rr - t)
        circ_x = (rr + dr_c) * np.cos(phi)
        circ_y = (rr + dr_c) * np.sin(phi)
        ax.plot(circ_x, circ_y, np.zeros_like(phi),
                color='#1a6bbf', alpha=0.35, linewidth=0.6)
    style_ax(ax, zlim=1.4)

    # ── G-MODE (riga 1) ──────────────────────────────────────
    ax = axes[1][col]
    Xg, Yg, Zu, Zl, drho_g = g_mode(t)
    
    fc_up  = CMAP(NORM( drho_g))
    fc_low = CMAP(NORM(-drho_g))

    draw_midplane(ax)

    # superficie inferiore
    ax.plot_surface(Xg, Yg, Zl,
                    facecolors=fc_low,
                    rstride=2, cstride=2,
                    alpha=0.88, shade=True)

    # superficie superiore
    ax.plot_surface(Xg, Yg, Zu,
                    facecolors=fc_up,
                    rstride=2, cstride=2,
                    alpha=0.88, shade=True)

    # frecce verticali su una corona radiale
    phi_arr = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    for rr in [5.5, 7.2]:
        env_r = np.exp(-((rr - r0)**2) / (2 * sigma**2))
        dz    = 0.7 * H_OVER_R * rr * env_r * np.cos(k_g * rr) * np.cos(t)
        for pa in phi_arr:
            xa, ya = rr * np.cos(pa), rr * np.sin(pa)
            ax.quiver(xa, ya,  0.05,  0, 0,  dz * 0.8,
                      color='#b07800', alpha=0.65, linewidth=0.7,
                      arrow_length_ratio=0.35)
            ax.quiver(xa, ya, -0.05,  0, 0, -dz * 0.8,
                      color='#cc4400', alpha=0.65, linewidth=0.7,
                      arrow_length_ratio=0.35)
    style_ax(ax, zlim=1.6)

    # ── C-MODE (riga 2) ──────────────────────────────────────
    ax = axes[2][col]
    Xc, Yc, Zc, drho_c = c_mode(t)
    fc_c = CMAP(NORM(Zc / (H_OVER_R * R_OUT * 1.1)))
    draw_midplane(ax)
    ax.plot_surface(Xc, Yc, Zc,
                    facecolors=fc_c,
                    rstride=2, cstride=2,
                    alpha=0.88, shade=True)
    # anelli che seguono la corrugazione
    for rr in np.linspace(R_TRAP_IN + 0.3, R_TRAP_OUT - 0.3, 6):
        env_r = np.exp(-((rr - r0)**2) / (2 * sigma**2))
        zz    = H_OVER_R * rr * env_r * np.cos(phi - t - np.pi/2) \
                * np.cos(k_c * (rr - r0))
        ax.plot(rr * np.cos(phi), rr * np.sin(phi), zz,
                color='#1a8a1a', alpha=0.40, linewidth=0.7)
    # asse di precessione (linea di nodo)
    theta_node = t + np.pi/2
    ax.plot([0, R_OUT * np.cos(theta_node)],
            [0, R_OUT * np.sin(theta_node)],
            [0, 0],
            color='#1a8a1a', alpha=0.6, linewidth=0.8,
            linestyle='--')
    ax.plot([0, -R_OUT * np.cos(theta_node)],
            [0, -R_OUT * np.sin(theta_node)],
            [0, 0],
            color='#1a8a1a', alpha=0.6, linewidth=0.8,
            linestyle='--')
    style_ax(ax, zlim=1.6)

# ──────────────────────────────────────────────────────────────
# ETICHETTE COLONNE (snapshot temporali) — in cima
# ──────────────────────────────────────────────────────────────
for col, tlab in enumerate(T_LABELS):
    ax_top = axes[0][col]
    ax_top.set_title(tlab, color='#222222',
                     pad=4, fontstyle='italic')

# ──────────────────────────────────────────────────────────────
# ETICHETTE RIGHE (nomi modi) — a sinistra, usando text su fig
# ──────────────────────────────────────────────────────────────
# posizioni y approssimate dei centri delle tre righe
row_y_positions = [0.77, 0.50, 0.23]

for row, (title, color, ypos) in enumerate(
        zip(ROW_TITLES, ROW_COLORS, row_y_positions)):
    fig.text(
        0.02, ypos, title,
        color=color,
        fontweight='bold', fontstyle='italic',
        ha='center', va='center',
        rotation=90
    )

# ──────────────────────────────────────────────────────────────
# COLORBAR
# ──────────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.905, 0.12, 0.015, 0.72])
sm = ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r'$\delta\rho\,/\,\rho_0$  (normalized)',
            color='#333333', labelpad=8)
cbar.ax.yaxis.set_tick_params(color='#333333', labelsize=6)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#333333')
cbar.outline.set_edgecolor('#aaaaaa')
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

# ──────────────────────────────────────────────────────────────
# TITOLO E CAPTION
# ──────────────────────────────────────────────────────────────
"""
fig.suptitle(
    "Oscillation modes of a geometrically thin relativistic accretion disk",
     color='#d8d0be', y=0.975, fontstyle='italic'
)

caption = (
    Fig. X — Three snapshots ($\omega t = 0,\,\pi/2,\,\pi$) of the fundamental oscillation modes.
    \textbf{Top:} P-mode ($n{=}0,\,m{=}0$): inertial–acoustic wave;
    the disk plane remains flat while density $\delta\rho/\rho_0$ oscillates radially.
    \textbf{Middle:} G-mode ($n{=}1,\,m{=}0$): inertial–gravity wave;
    upper (warm) and lower (cool) layers oscillate in antiphase.
    \textbf{Bottom:} C-mode ($n{=}1,\,m{=}1$): corrugation wave;
    the disk plane itself is warped with a one-armed ($m{=}1$) pattern
    precessing at $\omega\sim\Omega_\mathrm{LT}\ll\kappa_\mathrm{max}$.
    Dashed lines mark the instantaneous line of nodes.
    All modes are confined to the trapping region
    ${R_TRAP_IN:.1f}\leq r/r_g\leq{R_TRAP_OUT:.1f}$
    by the relativistic epicyclic barrier.
    Color encodes the normalized density perturbation.
)

# Se usetex=False usiamo la versione senza comandi LaTeX
caption_plain = (
    f"Fig. X — Three snapshots (ωt = 0, π/2, π) of the fundamental oscillation modes. "
    f"Top: P-mode (n=0, m=0): inertial–acoustic wave; the disk plane remains flat "
    f"while density δρ/ρ₀ oscillates radially. "
    f"Middle: G-mode (n=1, m=0): inertial–gravity wave; upper and lower layers oscillate in antiphase. "
    f"Bottom: C-mode (n=1, m=1): corrugation wave; the disk plane is warped with a one-armed (m=1) pattern "
    f"precessing at ω ~ Ω_LT ≪ κ_max. Dashed lines mark the line of nodes. "
    f"All modes are trapped in {R_TRAP_IN:.1f} ≤ r/rg ≤ {R_TRAP_OUT:.1f}. "
    f"Color encodes the normalized density perturbation."
)

fig.text(
    0.5, 0.005, caption_plain,
    ha='center', va='bottom',
    , color='#7a7a9a',
    wrap=True,
    style='italic',
    transform=fig.transFigure
)
"""

# ──────────────────────────────────────────────────────────────
# SALVATAGGIO
# ──────────────────────────────────────────────────────────────
plt.savefig(
    "cmodes_plots/disk_modes_snapshots.pdf",
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.savefig(
    "cmodes_plots/disk_modes_snapshots.png",
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)

print("Salvato: disk_modes_snapshots.pdf")
print("Salvato: disk_modes_snapshots.png  (300 dpi)")
plt.show()