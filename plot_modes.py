"""
Disk Oscillation Modes — Single Row (t=0 snapshot)
====================================================
Tre pannelli in riga orizzontale: p-mode, g-mode, c-mode al tempo t=0.
Output: disk_modes_row.pdf  +  disk_modes_row.png (300 dpi)

Dipendenze: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.colors import Normalize, LightSource
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
from setup import set_style_beamer, fix_spines

# ──────────────────────────────────────────────────────────────
# PARAMETRI (identici all'originale)
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
# FUNZIONI DEI MODI (identiche all'originale)
# ──────────────────────────────────────────────────────────────

def p_mode(t, A=0.9):
    spatial = ENV * np.cos(k_p * R)
    dr    = 0.18 * A * spatial * np.cos(t)
    drho  = A * spatial * np.cos(t)
    X     = (R + dr) * np.cos(PHI)
    Y     = (R + dr) * np.sin(PHI)
    Z     = np.zeros_like(R)
    return X, Y, Z, drho

def g_mode(t, A=1.0):
    radial   = np.cos(k_g * R)
    amp      = A * H_loc * ENV * radial
    Z_upper  =  amp
    Z_lower  = -amp
    drho     = A * ENV * radial * (np.cos(t))**2
    return X0, Y0, Z_upper, Z_lower, drho

def c_mode(t, A=1.0):
    Z    = A * H_loc * ENV * np.cos(PHI - t - np.pi/2) * np.cos(k_c * (R - r0))
    drho = 0.35 * A * ENV * np.cos(PHI - t - np.pi/2) * np.cos(k_c * (R - r0))
    return X0, Y0, Z, drho

# ──────────────────────────────────────────────────────────────
# STILE
# ──────────────────────────────────────────────────────────────
set_style_beamer()
plt.rcParams.update({"figure.figsize": (5.0, 2.0)})  # override solo figsize

CMAP     = cm.RdBu_r
NORM     = Normalize(vmin=-1, vmax=1)
BG       = 'white'
GRID_COL = '#cccccc'
PANE_COL = 'white'

MODE_COLORS = ['#1a6bbf', '#b07800', '#1a8a1a']
MODE_TITLES = [
    r"(0,0) p-mode",
    r"(0,1) g-mode",
    r"(1,1) c-mode",
]

ELEV, AZIM = 24, -52
t = 0.0   # solo snapshot iniziale

# ──────────────────────────────────────────────────────────────
# FIGURA: 1 riga × 3 colonne + colorbar
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(5, 2), facecolor='white')

gs = gridspec.GridSpec(
    1, 4,
    figure=fig,
    left=0.02, right=0.91,
    top=0.88, bottom=0.05,
    wspace=0.02,
    width_ratios=[1, 1, 1, 0.06]
)

ax_p = fig.add_subplot(gs[0, 0], projection='3d')
ax_g = fig.add_subplot(gs[0, 1], projection='3d')
ax_c = fig.add_subplot(gs[0, 2], projection='3d')
cbar_ax = fig.add_axes([0.925, 0.12, 0.015, 0.70])

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
    ax.set_xlabel(r'$x\,[r_g]$', color='#333333', labelpad=-7)
    ax.set_ylabel(r'$y\,[r_g]$', color='#333333', labelpad=-7)
    ax.set_zlabel(r'$z/H$', fontsize=7, color='#333333', labelpad=-9)
    ax.view_init(elev=ELEV, azim=AZIM)

def draw_midplane(ax):
    ax.plot_wireframe(
        X0, Y0, np.zeros_like(X0),
        rstride=12, cstride=12,
        color='#aaaaaa', alpha=0.45, linewidth=0.25
    )

# ── P-MODE ──────────────────────────────────────────────────
Xp, Yp, Zp, drho_p = p_mode(t)
fc_p = CMAP(NORM(drho_p))
draw_midplane(ax_p)
ax_p.plot_surface(Xp, Yp, Zp,
                  facecolors=fc_p,
                  rstride=2, cstride=2,
                  alpha=0.88, shade=True,
                  rasterized=True)
for rr in np.linspace(R_TRAP_IN + 0.5, R_TRAP_OUT - 0.5, 4):
    env_r = np.exp(-((rr - r0)**2) / (2 * sigma**2))
    dr_c  = 0.18 * env_r * np.cos(k_p * rr - t)
    circ_x = (rr + dr_c) * np.cos(phi)
    circ_y = (rr + dr_c) * np.sin(phi)
    ax_p.plot(circ_x, circ_y, np.zeros_like(phi),
              color='#1a6bbf', alpha=0.35, linewidth=0.6)
style_ax(ax_p, zlim=1.4)
ax_p.set_title(MODE_TITLES[0], color=MODE_COLORS[0],
               fontweight='bold', fontstyle='italic', pad=4)

# ── G-MODE ──────────────────────────────────────────────────
Xg, Yg, Zu, Zl, drho_g = g_mode(t)
fc_up  = CMAP(NORM( drho_g))
fc_low = CMAP(NORM(-drho_g))
draw_midplane(ax_g)
ax_g.plot_surface(Xg, Yg, Zl,
                  facecolors=fc_low,
                  rstride=2, cstride=2,
                  alpha=0.88, shade=True,
                  rasterized=True)
ax_g.plot_surface(Xg, Yg, Zu,
                  facecolors=fc_up,
                  rstride=2, cstride=2,
                  alpha=0.88, shade=True,
                  rasterized=True)
phi_arr = np.linspace(0, 2 * np.pi, 10, endpoint=False)
for rr in [5.5, 7.2]:
    env_r = np.exp(-((rr - r0)**2) / (2 * sigma**2))
    dz    = 0.7 * H_OVER_R * rr * env_r * np.cos(k_g * rr) * np.cos(t)
    for pa in phi_arr:
        xa, ya = rr * np.cos(pa), rr * np.sin(pa)
        ax_g.quiver(xa, ya,  0.05,  0, 0,  dz * 0.8,
                    color='#b07800', alpha=0.65, linewidth=0.7,
                    arrow_length_ratio=0.35)
        ax_g.quiver(xa, ya, -0.05,  0, 0, -dz * 0.8,
                    color='#cc4400', alpha=0.65, linewidth=0.7,
                    arrow_length_ratio=0.35)
style_ax(ax_g, zlim=1.6)
ax_g.set_title(MODE_TITLES[1], color=MODE_COLORS[1],
               fontweight='bold', fontstyle='italic', pad=4)

# ── C-MODE ──────────────────────────────────────────────────
Xc, Yc, Zc, drho_c = c_mode(t)
fc_c = CMAP(NORM(Zc / (H_OVER_R * R_OUT * 1.1)))
draw_midplane(ax_c)
ax_c.plot_surface(Xc, Yc, Zc,
                  facecolors=fc_c,
                  rstride=2, cstride=2,
                  alpha=0.88, shade=True,
                  rasterized=True)
for rr in np.linspace(R_TRAP_IN + 0.3, R_TRAP_OUT - 0.3, 6):
    env_r = np.exp(-((rr - r0)**2) / (2 * sigma**2))
    zz    = H_OVER_R * rr * env_r * np.cos(phi - t - np.pi/2) \
            * np.cos(k_c * (rr - r0))
    ax_c.plot(rr * np.cos(phi), rr * np.sin(phi), zz,
              color='#1a8a1a', alpha=0.40, linewidth=0.7)
theta_node = t + np.pi/2
ax_c.plot([0, R_OUT * np.cos(theta_node)],
          [0, R_OUT * np.sin(theta_node)],
          [0, 0],
          color='#1a8a1a', alpha=0.6, linewidth=0.8, linestyle='--')
ax_c.plot([0, -R_OUT * np.cos(theta_node)],
          [0, -R_OUT * np.sin(theta_node)],
          [0, 0],
          color='#1a8a1a', alpha=0.6, linewidth=0.8, linestyle='--')
style_ax(ax_c, zlim=1.6)
ax_c.set_title(MODE_TITLES[2], color=MODE_COLORS[2],
               fontweight='bold', fontstyle='italic', pad=4)

# ── COLORBAR ────────────────────────────────────────────────
sm = ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(r'$\delta\rho\,/\,\rho_0$',
               color='#333333', labelpad=8)
cbar.ax.yaxis.set_tick_params(color='#333333', labelsize=6)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#333333')
cbar.outline.set_edgecolor('#aaaaaa')
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

# ──────────────────────────────────────────────────────────────
# SALVATAGGIO
# ──────────────────────────────────────────────────────────────
plt.savefig(
    "presentazione/disk_modes_row.pdf",
    bbox_inches='tight',
    facecolor='white'
)

print("Salvato: disk_modes_row.pdf")
plt.show()