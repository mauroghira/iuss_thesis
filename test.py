import numpy as np
import matplotlib.pyplot as plt
import time

# --------------------------------------------------------
# FREQUENCIES
# --------------------------------------------------------

def nu_phi(R, a):
    return 1.0 / (R**1.5 + abs(a))

def nu_theta(R, a):
    return nu_phi(R, a) * np.sqrt(1 - 4*abs(a)/R**1.5 + 3*a**2/R**2)

def nu_LT(R, a):
    return nu_phi(R, a) - nu_theta(R, a)


# --------------------------------------------------------
# NUMERIC TRUE INTEGRATION
# --------------------------------------------------------

def nu_prec_numeric(a, r_in, r_out, zeta, n_rad=50000):
    R = np.linspace(r_in, r_out, n_rad)
    Sigma = R**(-zeta)
    weight = Sigma * R**3 * nu_phi(R, a)

    num = np.trapezoid(nu_LT(R, a) * weight, R)
    den = np.trapezoid(weight, R)

    return num / den


# --------------------------------------------------------
# VECTORIZED APPROXIMATE VERSION
# (fixed version that works for scalars)
# --------------------------------------------------------

def nu_prec_vectorized(a, rin, rout, zeta, n_rad=2000):
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

# se quella sopra da problemi usa questa, ma la sopra è moto pi+velopce
def nu_prec_numeric_vectorized(a, r_in, r_out, zeta, n_rad=20000):
    """
    Fully numeric, non-approximated integration — vectorized.
    a, r_in, r_out, zeta can be arrays, all broadcastable.
    Integration variable is R, but rewritten onto a normalized x-grid.
    """
    # Ensure arrays
    a = np.asarray(a)
    r_in = np.asarray(r_in)
    r_out = np.asarray(r_out)
    zeta = np.asarray(zeta)

    # Normalized radial grid
    x = np.linspace(0, 1, n_rad)
    x = x.reshape((1,)*a.ndim + (n_rad,))

    # Physical radius grid
    R = r_in[...,None] + x * (r_out - r_in)[...,None]  # shape = broadcast + (n_rad,)

    # Compute integrand
    Sigma = R**(-zeta[...,None])
    weight = Sigma * R**3 * nu_phi(R, a[...,None])

    integrand = nu_LT(R, a[...,None]) * weight

    # dR = (r_out - r_in) * dx
    dR = (r_out - r_in)[...,None]

    # integrate along x axis
    num = np.trapezoid(integrand, x, axis=-1) * dR
    den = np.trapezoid(weight,    x, axis=-1) * dR

    return num / den


# --------------------------------------------------------
# BENCHMARK PARAMETERS
# --------------------------------------------------------

a     = [-0.998, -0.5, 0, 0.5, 0.998]  # test multiple spins
r_in  = 6.0
zeta  = [-1, -0.5, 0, 0.5, 1]  # test multiple surface density profiles

# scan over outer radius
r_out_vals = np.linspace(1.05*r_in, 200, 200)


# --------------------------------------------------------
# RUN BENCHMARK
# --------------------------------------------------------

nu_true   = np.zeros_like(r_out_vals)
#nu_approx = np.zeros_like(r_out_vals)

t_num = []
t_vec = []
t_vec_2 = []

for i, r_out in enumerate(r_out_vals):

    # numeric
    t0 = time.perf_counter()
    for aa in a:
        for z in zeta:
            nu_true[i] = nu_prec_numeric(aa, r_in, r_out, z)
    t_num.append(time.perf_counter() - t0)

    # vectorized
    t0 = time.perf_counter()
    nu_approx = nu_prec_vectorized(a, r_in, r_out, zeta)
    t_vec.append(time.perf_counter() - t0)

    # vectorized
    t0 = time.perf_counter()
    nu_approx2 = nu_prec_numeric_vectorized(a, r_in, r_out, zeta)
    t_vec_2.append(time.perf_counter() - t0)


# --------------------------------------------------------
# COMPUTE ERROR
# --------------------------------------------------------

#rel_err = np.abs(nu_true - nu_approx) / nu_true


# --------------------------------------------------------
# PRINT SPEEDUP
# --------------------------------------------------------

print("\nBenchmark completed")
print(f"Average numeric time:     {np.mean(t_num):.4f} s")
print(f"Average vectorized time:  {np.mean(t_vec):.4f} s")
print(f"Speed-up factor:          {np.mean(t_num)/np.mean(t_vec):.1f}x")
print(f"Average vectorized time:  {np.mean(t_vec_2):.4f} s")
print(f"Speed-up factor (num vec):{np.mean(t_num)/np.mean(t_vec_2):.1f}x")


# --------------------------------------------------------
# PLOTS
# --------------------------------------------------------
"""
plt.figure(figsize=(8,6))
plt.loglog(r_out_vals, nu_true, label="Numeric (true)", lw=2)
plt.loglog(r_out_vals, nu_approx, "--", label="Vectorized approx", lw=2)
plt.xlabel("r_out")
plt.ylabel("ν_prec  (geometric units)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Precession frequency: true vs. vectorized")
plt.show()

plt.figure(figsize=(8,6))
plt.loglog(r_out_vals, rel_err)
plt.xlabel("r_out")
plt.ylabel("Relative error")
plt.grid(True, alpha=0.3)
plt.title("Relative error between numeric and vectorized")
plt.show()
"""
plt.figure(figsize=(8,6))
plt.plot(r_out_vals, t_num, label="numeric")
plt.plot(r_out_vals, t_vec, label="vectorized")
plt.xlabel("r_out")
plt.ylabel("seconds per evaluation")
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Timing per integration")
plt.show()
