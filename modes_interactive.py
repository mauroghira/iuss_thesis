"""
3D Visualization of Accretion Disk Oscillation Modes
=====================================================
Visualizza i tre modi fondamentali di oscillazione di un disco di accrescimento:
  - P-mode (n=0): onda inerziale-acustica radiale, disco piatto che pulsa
  - G-mode (n=1): onda di gravità interna, strati verticali in opposizione di fase
  - C-mode (n=1, m=1): onda di corrugazione, piano del disco ondulato che precede

Uso:
    python disk_modes_3d.py

Dipendenze:
    pip install numpy matplotlib
"""

import matplotlib
matplotlib.use('Qt5Agg')   # prova TkAgg prima; se fallisce vedi nota sotto
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# ─────────────────────────────────────────────
# PARAMETRI FISICI DEL DISCO
# ─────────────────────────────────────────────
R_IN   = 3.0    # raggio interno (unità di r_g)
R_OUT  = 12.0   # raggio esterno
H_OVER_R = 0.1  # aspect ratio H/r (geometricamente sottile)

# Regione di trapping (dove il modo è confinato)
R_TRAP_IN  = 4.0
R_TRAP_OUT = 9.0

Nr   = 80
Nphi = 120

r   = np.linspace(R_IN, R_OUT, Nr)
phi = np.linspace(0, 2 * np.pi, Nphi)
R, PHI = np.meshgrid(r, phi)

X_base = R * np.cos(PHI)
Y_base = R * np.sin(PHI)

# ─────────────────────────────────────────────
# PROFILO RADIALE DI INTRAPPOLAMENTO
# Gaussiana centrata nella regione trapped
# ─────────────────────────────────────────────
r0    = 0.5 * (R_TRAP_IN + R_TRAP_OUT)
sigma = 0.5 * (R_TRAP_OUT - R_TRAP_IN) / 2.0

def radial_envelope(R):
    """Inviluppo gaussiano che simula il trapping."""
    env = np.exp(-((R - r0) ** 2) / (2 * sigma ** 2))
    # azzera fuori dalla regione di trapping
    env[R < R_TRAP_IN]  *= np.exp(-((R[R < R_TRAP_IN]  - R_TRAP_IN)  ** 2) / (0.3 ** 2))
    env[R > R_TRAP_OUT] *= np.exp(-((R[R > R_TRAP_OUT] - R_TRAP_OUT) ** 2) / (0.3 ** 2))
    return env

ENVELOPE = radial_envelope(R)

# Spessore locale del disco H(r)
H_local = H_OVER_R * R

# ─────────────────────────────────────────────
# DEFINIZIONE DEI TRE MODI
# ─────────────────────────────────────────────

def p_mode(R, PHI, t, m=0, omega=1.0, A=1.0):
    """
    P-mode (n=0, m=0):
    Onda inerziale-acustica radiale. Il disco rimane piatto (Z~0),
    ma la densità/pressione oscilla radialmente.
    Visualizzato come deformazione radiale lieve del piano + colore.

    Struttura: cos(k_r * r - omega * t)
    """
    k_r = 2.0 / (R_TRAP_OUT - R_TRAP_IN) * np.pi  # ~1 lunghezza d'onda nella regione
    
    # Perturbazione di densità (modulazione del colore)
    delta_rho = A * ENVELOPE * np.cos(k_r * R - omega * t + m * PHI)
    
    # Piccola deformazione radiale (il disco si "gonfia" radialmente)
    delta_r = 0.15 * A * ENVELOPE * np.cos(k_r * R - omega * t + m * PHI)
    
    X = (R + delta_r) * np.cos(PHI)
    Y = (R + delta_r) * np.sin(PHI)
    Z = np.zeros_like(R)  # piano piatto
    
    return X, Y, Z, delta_rho


def g_mode(R, PHI, t, m=0, omega=0.8, A=1.0):
    """
    G-mode (n=1, m=0):
    Onda di gravità interna. Struttura verticale antisimmetrica: H_1(eta) ~ eta.
    Gli strati sopra e sotto il piano equatoriale oscillano in OPPOSIZIONE di fase.
    
    Visualizzato come due superfici (strato superiore e inferiore) che si deformano
    in direzioni opposte.
    """
    k_r = 1.5 * np.pi / (R_TRAP_OUT - R_TRAP_IN)
    
    # Spostamento verticale: proporzionale a eta = z/H
    # La struttura verticale è H_1(eta) ~ eta, quindi il moto è antisimmetrico
    # Mostriamo lo strato a eta = +1 (superficie superiore del disco)
    xi_z_upper = A * H_local * ENVELOPE * np.cos(k_r * R - omega * t + m * PHI)
    xi_z_lower = -xi_z_upper  # OPPOSIZIONE DI FASE
    
    # Perturbazione di densità (antisimmetrica verticalmente)
    delta_rho = A * ENVELOPE * np.cos(k_r * R - omega * t + m * PHI)
    
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    
    return X, Y, xi_z_upper, xi_z_lower, delta_rho


def c_mode(R, PHI, t, m=1, omega=0.05, A=1.0):
    """
    C-mode (n=1, m=1):
    Onda di corrugazione. Il piano del disco viene INCLINATO con pattern a m bracci.
    Struttura verticale H_1(eta) ~ eta (come g-mode), ma la quantità primaria
    è lo SPOSTAMENTO xi_z del piano, non la velocità u_z.
    
    Frequenza molto bassa: omega ~ Omega_LT << kappa_max
    Il piano ondulato precede lentamente.
    """
    # Pattern spaziale: m bracci azimutali + struttura radiale
    k_r = 0.8 * np.pi / (R_TRAP_OUT - R_TRAP_IN)
    
    # Lo spostamento del piano è la quantità dominante
    xi_z = A * H_local * ENVELOPE * np.cos(m * PHI - omega * t) * np.cos(k_r * (R - r0))
    
    # Piccola perturbazione di densità (secondaria nel c-mode)
    delta_rho = 0.3 * A * ENVELOPE * np.cos(m * PHI - omega * t) * np.cos(k_r * (R - r0))
    
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    Z = xi_z
    
    return X, Y, Z, delta_rho


# ─────────────────────────────────────────────
# SETUP FIGURA
# ─────────────────────────────────────────────

plt.rcParams.update({
    'figure.facecolor': '#0a0a14',
    'axes.facecolor':   '#0a0a14',
    'text.color':       '#e8e4d8',
    'axes.labelcolor':  '#e8e4d8',
    'xtick.color':      '#e8e4d8',
    'ytick.color':      '#e8e4d8',
    'font.family':      'monospace',
})

fig = plt.figure(figsize=(18, 11), facecolor='#0a0a14')
fig.suptitle(
    "DISK OSCILLATION MODES  ·  Thin Relativistic Accretion Disk",
    fontsize=13, color='#c8b89a', fontfamily='monospace', y=0.97,
    fontweight='bold'
)

# Layout: 3 plot 3D in alto, pannello info + slider in basso
gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    top=0.90, bottom=0.18,
    left=0.04, right=0.98,
    hspace=0.05, wspace=0.05
)

ax_p = fig.add_subplot(gs[0, 0], projection='3d')
ax_g = fig.add_subplot(gs[0, 1], projection='3d')
ax_c = fig.add_subplot(gs[0, 2], projection='3d')

# Pannello testo informativo
ax_info = fig.add_axes([0.04, 0.02, 0.92, 0.14], facecolor='#0f0f1e')
ax_info.axis('off')

# Slider per il tempo
ax_slider = fig.add_axes([0.15, 0.015, 0.7, 0.025], facecolor='#1a1a2e')
slider_t = Slider(
    ax_slider, 'Phase  ωt', 0.0, 2 * np.pi,
    valinit=0.0, color='#4a6fa5',
    initcolor='none'
)
slider_t.label.set_color('#c8b89a')
slider_t.valtext.set_color('#c8b89a')


# ─────────────────────────────────────────────
# STILE ASSI 3D
# ─────────────────────────────────────────────

def style_3d_ax(ax, title, subtitle, color_title):
    ax.set_facecolor('#0a0a14')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1e1e3a')
    ax.yaxis.pane.set_edgecolor('#1e1e3a')
    ax.zaxis.pane.set_edgecolor('#1e1e3a')
    ax.grid(True, color='#1e1e3a', linewidth=0.4, alpha=0.6)
    ax.set_xlabel('x  [rg]', fontsize=7, color='#6a6a8a', labelpad=-8)
    ax.set_ylabel('y  [rg]', fontsize=7, color='#6a6a8a', labelpad=-8)
    ax.set_zlabel('z / H',   fontsize=7, color='#6a6a8a', labelpad=-6)
    ax.tick_params(colors='#3a3a5a', labelsize=6, pad=-3)
    ax.set_title(
        f"{title}\n{subtitle}",
        fontsize=10, color=color_title,
        fontfamily='monospace', pad=4,
        fontweight='bold'
    )
    ax.view_init(elev=22, azim=-55)


# ─────────────────────────────────────────────
# DISEGNO DEL DISCO DI RIFERIMENTO (wireframe base)
# ─────────────────────────────────────────────

def draw_reference_disk(ax, alpha=0.08):
    """Disco di riferimento non perturbato."""
    ax.plot_wireframe(
        X_base, Y_base, np.zeros_like(X_base),
        rstride=10, cstride=10,
        color='#2a2a4a', alpha=alpha, linewidth=0.3
    )


# ─────────────────────────────────────────────
# FUNZIONE DI AGGIORNAMENTO
# ─────────────────────────────────────────────

cmap_density = cm.RdBu_r
cmap_corrugation = cm.coolwarm

def update(val):
    t = slider_t.val

    # ── P-MODE ──────────────────────────────
    ax_p.cla()
    style_3d_ax(ax_p,
                "P-MODE  (n=0, m=0)",
                "inertial–acoustic  ·  radial compression",
                '#7eb8f7')

    Xp, Yp, Zp, drho_p = p_mode(R, PHI, t, A=0.8)
    
    norm_p = Normalize(vmin=-1, vmax=1)
    colors_p = cmap_density(norm_p(drho_p))
    
    draw_reference_disk(ax_p)
    ax_p.plot_surface(
        Xp, Yp, Zp,
        facecolors=colors_p,
        rstride=2, cstride=2,
        alpha=0.85, shade=True,
        lightsource=None
    )
    # Aggiungi frecce radiali per indicare il moto
    r_arrows = np.array([5.0, 6.5, 8.0])
    phi_arrows = np.linspace(0, 2*np.pi, 8, endpoint=False)
    k_r_arr = 2.0 / (R_TRAP_OUT - R_TRAP_IN) * np.pi
    for ra in r_arrows:
        env_a = np.exp(-((ra - r0)**2) / (2*sigma**2))
        for pa in phi_arrows:
            dr = 0.3 * env_a * np.cos(k_r_arr * ra - 1.0 * t)
            xa, ya = ra * np.cos(pa), ra * np.sin(pa)
            dxa = dr * np.cos(pa)
            dya = dr * np.sin(pa)
            ax_p.quiver(xa, ya, 0, dxa, dya, 0,
                       length=0.6, color='#7eb8f7', alpha=0.5,
                       arrow_length_ratio=0.3, linewidth=0.7)

    ax_p.set_zlim(-1.5, 1.5)
    ax_p.set_xlim(-R_OUT, R_OUT)
    ax_p.set_ylim(-R_OUT, R_OUT)

    # ── G-MODE ──────────────────────────────
    ax_g.cla()
    style_3d_ax(ax_g,
                "G-MODE  (n=1, m=0)",
                "inertial–gravity  ·  vertical breathing",
                '#f7c97e')

    Xg, Yg, Z_upper, Z_lower, drho_g = g_mode(R, PHI, t, A=1.0)
    
    norm_g = Normalize(vmin=-1, vmax=1)
    colors_g_up  = cmap_density(norm_g( drho_g))
    colors_g_low = cmap_density(norm_g(-drho_g))  # opposta
    
    draw_reference_disk(ax_g)
    # Superficie superiore
    ax_g.plot_surface(
        Xg, Yg, Z_upper,
        facecolors=colors_g_up,
        rstride=2, cstride=2,
        alpha=0.75, shade=True
    )
    # Superficie inferiore (in opposizione di fase)
    ax_g.plot_surface(
        Xg, Yg, Z_lower,
        facecolors=colors_g_low,
        rstride=2, cstride=2,
        alpha=0.75, shade=True
    )
    # Frecce verticali per indicare l'opposizione di fase
    r_arrows_g = np.array([5.5, 7.0, 8.5])
    phi_arrows_g = np.linspace(np.pi/6, 2*np.pi, 6, endpoint=False)
    k_r_g = 1.5 * np.pi / (R_TRAP_OUT - R_TRAP_IN)
    for ra in r_arrows_g:
        env_a = np.exp(-((ra - r0)**2) / (2*sigma**2))
        for pa in phi_arrows_g:
            dz = 0.5 * H_OVER_R * ra * env_a * np.cos(k_r_g * ra - 0.8 * t)
            xa, ya = ra * np.cos(pa), ra * np.sin(pa)
            ax_g.quiver(xa, ya,  0.05,  0, 0,  dz,
                       length=0.5, color='#f7c97e', alpha=0.55,
                       arrow_length_ratio=0.35, linewidth=0.7)
            ax_g.quiver(xa, ya, -0.05,  0, 0, -dz,
                       length=0.5, color='#f0906a', alpha=0.55,
                       arrow_length_ratio=0.35, linewidth=0.7)

    ax_g.set_zlim(-1.5, 1.5)
    ax_g.set_xlim(-R_OUT, R_OUT)
    ax_g.set_ylim(-R_OUT, R_OUT)

    # ── C-MODE ──────────────────────────────
    ax_c.cla()
    style_3d_ax(ax_c,
                "C-MODE  (n=1, m=1)",
                "corrugation wave  ·  disk plane warping",
                '#a8f7a0')

    Xc, Yc, Zc, drho_c = c_mode(R, PHI, t, A=1.0)
    
    norm_c = Normalize(vmin=-1, vmax=1)
    colors_c = cmap_corrugation(norm_c(Zc / (H_OVER_R * R_OUT)))
    
    draw_reference_disk(ax_c, alpha=0.05)
    ax_c.plot_surface(
        Xc, Yc, Zc,
        facecolors=colors_c,
        rstride=2, cstride=2,
        alpha=0.88, shade=True
    )
    # Linea di nodo (dove Z=0): evidenzia la struttura a m=1
    phi_line = np.linspace(0, 2*np.pi, 300)
    for rr in np.linspace(R_TRAP_IN, R_TRAP_OUT, 5):
        env_a = np.exp(-((rr - r0)**2) / (2*sigma**2))
        k_r_c = 0.8 * np.pi / (R_TRAP_OUT - R_TRAP_IN)
        zz = H_OVER_R * rr * env_a * np.cos(phi_line - 0.05 * t) * np.cos(k_r_c * (rr - r0))
        xx = rr * np.cos(phi_line)
        yy = rr * np.sin(phi_line)
        ax_c.plot(xx, yy, zz, color='#a8f7a0', alpha=0.35, linewidth=0.8)

    ax_c.set_zlim(-1.5, 1.5)
    ax_c.set_xlim(-R_OUT, R_OUT)
    ax_c.set_ylim(-R_OUT, R_OUT)

    # ── PANNELLO INFORMATIVO ────────────────
    ax_info.cla()
    ax_info.set_facecolor('#0f0f1e')
    ax_info.axis('off')

    info_text = (
        "  P-MODE (n=0, m=0)                    G-MODE (n=1, m=0)                    C-MODE (n=1, m=1)\n"
        "  ─────────────────────────────────     ──────────────────────────────────   ─────────────────────────────────────\n"
        "  Quantità primaria: δρ, δp             Quantità primaria: u_z , δρ           Quantità primaria: ξ_z  (spostamento)\n"
        "  Struttura vert.:   n=0  (piatto)      Struttura vert.:   H₁(η) ~ η          Struttura vert.:   H₁(η) ~ η\n"
        "  Simmetria:         piano equat.        Simmetria:         ANTISIMMETRICA       Simmetria:         m=1  bracci\n"
        "  Moto:              radiale  u_r         Moto:              verticale ↑↓ opp.   Moto:              ondulazione piano\n"
        "  Frequenza:         ω ~ κ_max            Frequenza:         ω ~ κ_max            Frequenza:         ω ~ Ω_LT << κ_max\n"
        "  Forza richiamo:    pressione            Forza richiamo:    gravità verticale    Forza richiamo:    gravità nel piano\n"
        "  Regione:           trapped (GR)         Regione:           trapped (GR)         Regione:           inner disk  (a>0)\n"
    )
    ax_info.text(
        0.01, 0.95, info_text,
        transform=ax_info.transAxes,
        fontsize=7.2, color='#b8b4a4',
        verticalalignment='top',
        fontfamily='monospace',
        linespacing=1.55
    )

    # Regione di trapping evidenziata
    trap_text = f"  Trapping region:  {R_TRAP_IN:.1f} – {R_TRAP_OUT:.1f} rg    |    Phase ωt = {t:.2f} rad"
    ax_info.text(
        0.01, 0.08, trap_text,
        transform=ax_info.transAxes,
        fontsize=7, color='#6a9ad4',
        fontfamily='monospace'
    )

    fig.canvas.draw_idle()


# ─────────────────────────────────────────────
# COLORBAR CONDIVISA
# ─────────────────────────────────────────────

cbar_ax = fig.add_axes([0.955, 0.20, 0.012, 0.65])
sm = cm.ScalarMappable(cmap=cmap_density, norm=Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('δρ / ρ₀  (normalized)', fontsize=7, color='#8a8a9a', labelpad=6)
cbar.ax.yaxis.set_tick_params(color='#8a8a9a', labelsize=6)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8a8a9a')
cbar.outline.set_edgecolor('#2a2a4a')

# ─────────────────────────────────────────────
# ANIMAZIONE AUTOMATICA
# ─────────────────────────────────────────────

from matplotlib.animation import FuncAnimation

def animate(frame):
    t_val = (frame / 60.0) * 2 * np.pi
    slider_t.set_val(t_val % (2 * np.pi))

slider_t.on_changed(update)

# Disegno iniziale
update(0)

print("╔══════════════════════════════════════════════════════════════╗")
print("║   DISK OSCILLATION MODES — Interactive 3D Visualization      ║")
print("╠══════════════════════════════════════════════════════════════╣")
print("║  Usa lo slider in basso per variare la fase ωt               ║")
print("║                                                               ║")
print("║  P-MODE  (blu)   — compressione radiale, disco piatto         ║")
print("║  G-MODE  (giallo)— oscillazione verticale antisimmetrica      ║")
print("║  C-MODE  (verde) — corrugazione del piano, m=1 braccio        ║")
print("║                                                               ║")
print("║  Frecce: direzione del moto perturbato                        ║")
print("║  Colore superficie: δρ/ρ₀  (rosso=compresso, blu=rarefatto)   ║")
print("╚══════════════════════════════════════════════════════════════╝")

# Avvia animazione automatica
ani = FuncAnimation(fig, animate, frames=120, interval=80, repeat=True)

plt.show()