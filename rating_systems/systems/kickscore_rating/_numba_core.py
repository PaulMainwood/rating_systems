"""Numba-accelerated core for KickScore (Constant + Matérn 3/2 kernel).

State-space model with 3D state vector [c, f, f'] where:
- c: constant component (captures stable baseline skill)
- f: Matérn 3/2 component (captures smooth time-varying form)
- f': derivative of Matérn component

Inference via EP with Kalman filter/RTS smoother per player.

Reference: Maystre, Kristof & Grossglauser, "Pairwise Comparisons with
Flexible Time-Dynamics", KDD 2019.  arXiv:1903.07746
"""

import math

import numpy as np
from numba import njit

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
_SQRT3 = math.sqrt(3.0)

# Gauss-Hermite quadrature for logistic moment matching (from GPML)
_LAMBDAS = np.array([0.44, 0.41, 0.40, 0.39, 0.36]) * _SQRT2
_CS = np.array([
    1.146480988574439e02,
    -1.508871030070582e03,
    2.676085036831241e03,
    -1.356294962039222e03,
    7.543285642111850e01,
])

# logphi coefficients (from GPML)
_LOGPHI_CS = np.array([
    4.82040000e-04, -1.42906000e-03, 1.32002432e-03, 9.46158903e-04,
    -4.55633398e-03, 5.56964649e-03, 1.25993962e-03, -1.62157538e-02,
    2.62965152e-02, -1.82976468e-03, -9.43951024e-02, 2.86135782e-01,
    1.00000000e+00, 1.00000000e+00,
])
_LOGPHI_RS = np.array([1.27536664, 5.01904973, 6.16020985, 7.40974061, 2.97886563])
_LOGPHI_QS = np.array([2.26052852, 9.39603402, 12.04895193, 17.08144075, 9.60896533, 3.36907521])


# ── Numerical utilities ──────────────────────────────────────────────


@njit(cache=True)
def _logphi(z):
    """Log of the standard normal CDF.  Returns (log Phi(z), Phi'(z)/Phi(z))."""
    if z * z < 0.0492:
        coef = -z / _SQRT2PI
        val = 0.0
        for c in _LOGPHI_CS:
            val = coef * (c + val)
        res = -2.0 * val - math.log(2.0)
        dres = math.exp(-(z * z) / 2.0 - res) / _SQRT2PI
    elif z < -11.3137:
        num = 0.5641895835477550741
        for r in _LOGPHI_RS:
            num = -z * num / _SQRT2 + r
        den = 1.0
        for q in _LOGPHI_QS:
            den = -z * den / _SQRT2 + q
        res = math.log(num / (2.0 * den)) - (z * z) / 2.0
        dres = abs(den / num) * math.sqrt(2.0 / math.pi)
    else:
        cdf = 0.5 * math.erfc(-z / _SQRT2)
        if cdf < 1e-300:
            cdf = 1e-300
        res = math.log(cdf)
        dres = math.exp(-(z * z) / 2.0 - res) / _SQRT2PI
    return res, dres


@njit(cache=True)
def _mm_probit_win(mean_cav, cov_cav):
    """Probit moment matching (from GPML likErf.m)."""
    z = mean_cav / math.sqrt(1.0 + cov_cav)
    logpart, val = _logphi(z)
    dlogpart = val / math.sqrt(1.0 + cov_cav)
    d2logpart = -val * (z + val) / (1.0 + cov_cav)
    return logpart, dlogpart, d2logpart


@njit(cache=True)
def _logsumexp2(xs, bs):
    a = xs[0]
    for i in range(1, len(xs)):
        if xs[i] > a:
            a = xs[i]
    s = 0.0
    for i in range(len(xs)):
        s += bs[i] * math.exp(xs[i] - a)
    if s <= 1e-30:
        s = 1e-30
    return a + math.log(s)


@njit(cache=True)
def mm_logit_win(mean_cav, cov_cav):
    """Logistic (Bradley-Terry) moment matching via scale mixture of probit.

    Returns (log Z, d log Z / dm, d² log Z / dm²) where Z is the
    normalisation constant of the tilted distribution.
    """
    arr1 = np.zeros(5)
    arr2 = np.zeros(5)
    arr3 = np.zeros(5)
    for i in range(5):
        x = _LAMBDAS[i]
        arr1[i], arr2[i], arr3[i] = _mm_probit_win(x * mean_cav, x * x * cov_cav)

    logpart1 = _logsumexp2(arr1, _CS)

    num_d = 0.0
    den_d = 0.0
    for i in range(5):
        w = math.exp(arr1[i]) * _CS[i]
        num_d += w * arr2[i] * _LAMBDAS[i]
        den_d += w
    if abs(den_d) < 1e-30:
        return 0.0, 0.0, 0.0
    dlogpart1 = num_d / den_d

    num_d2 = 0.0
    for i in range(5):
        w = math.exp(arr1[i]) * _CS[i]
        num_d2 += w * (arr2[i] * arr2[i] + arr3[i]) * _LAMBDAS[i] * _LAMBDAS[i]
    d2logpart1 = num_d2 / den_d - dlogpart1 * dlogpart1

    # Tail correction
    exponent = -10.0 * (abs(mean_cav) - (196.0 / 200.0) * cov_cav - 4.0)
    if exponent < 500.0:
        lambd = 1.0 / (1.0 + math.exp(exponent))
        logpart2 = min(cov_cav / 2.0 - abs(mean_cav), -0.1)
        dlogpart2 = 1.0
        if mean_cav > 0.0:
            logpart2 = math.log(1.0 - math.exp(logpart2))
            dlogpart2 = 0.0
        d2logpart2 = 0.0
    else:
        lambd = 0.0
        logpart2 = 0.0
        dlogpart2 = 0.0
        d2logpart2 = 0.0

    logpart = (1.0 - lambd) * logpart1 + lambd * logpart2
    dlogpart = (1.0 - lambd) * dlogpart1 + lambd * dlogpart2
    d2logpart = (1.0 - lambd) * d2logpart1 + lambd * d2logpart2
    return logpart, dlogpart, d2logpart


# ── Kernel matrices (Constant + Matérn 3/2, state dim = 3) ──────────


@njit(cache=True)
def compute_transition(dt, lambda_):
    """3x3 transition matrix for Constant + Matérn 3/2."""
    A = np.zeros((3, 3))
    # Constant block: A_c = [[1]]
    A[0, 0] = 1.0
    # Matérn 3/2 block
    a = lambda_
    e = math.exp(-a * dt)
    A[1, 1] = e * (a * dt + 1.0)
    A[1, 2] = e * dt
    A[2, 1] = e * (-a * a * dt)
    A[2, 2] = e * (1.0 - a * dt)
    return A


@njit(cache=True)
def compute_noise_cov(dt, var_m, lambda_):
    """3x3 noise covariance for Constant + Matérn 3/2."""
    Q = np.zeros((3, 3))
    # Constant block: Q_c = [[0]] (no noise)
    # Matérn 3/2 block
    a = lambda_
    da = dt * a
    c = math.exp(-2.0 * da)
    Q[1, 1] = var_m * (1.0 - c * (2.0 * da * da + 2.0 * da + 1.0))
    q12 = var_m * c * (2.0 * da * da * a)
    Q[1, 2] = q12
    Q[2, 1] = q12
    Q[2, 2] = var_m * a * a * (1.0 - c * (2.0 * da * da - 2.0 * da + 1.0))
    return Q


@njit(cache=True)
def compute_initial_cov(var_c, var_m, lambda_):
    """3x3 initial (stationary) covariance for Constant + Matérn 3/2."""
    P = np.zeros((3, 3))
    P[0, 0] = var_c
    P[1, 1] = var_m
    P[2, 2] = var_m * lambda_ * lambda_
    return P


# Measurement vector: observe c + f (not f')
_H = np.array([1.0, 1.0, 0.0])


# ── Kalman filter + RTS smoother ─────────────────────────────────────


@njit(cache=True)
def _solve3(A, B):
    """Solve A @ X = B for 3x3 A, 3x3 B, via the closed-form inverse.

    Chiefly here to drop the regularising jitter the LAPACK path needed: an
    absolute 1e-10 added to P_p is a ~1e-5 RELATIVE perturbation of the f'
    diagonal (var_m * lambda^2 ~ 9e-6 at lscale=365), which showed up as a
    ~1e-5 disagreement with the reference implementation that grew with the
    number of observations. P_p's condition number is only ~1e4, so the
    regularisation bought nothing.

    Speed was the original motivation and it is NOT the win: measured 1.1x on
    run_kalman, so the LAPACK dispatch was never the bottleneck. The real cost
    is per-step temporary allocation inside the Kalman loop (~8us/step across
    ~15 small 3x3 allocations); fixing that means scalarising the recursion or
    moving to a CSR layout with one numba call per sweep, as matern_ttt does.

    Cofactor inversion is less stable than LU in general, but A here is a
    predicted state covariance far from where float64 cofactors lose accuracy
    (verified: 6e-14 worst relative error vs LAPACK on realistic covariances).
    Reports failure when A is numerically singular so the caller can skip the
    update rather than propagate NaNs.
    """
    c00 = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    c01 = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    c02 = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    det = A[0, 0] * c00 + A[0, 1] * c01 + A[0, 2] * c02
    if abs(det) < 1e-300:
        return B, False

    c10 = A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2]
    c11 = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    c12 = A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1]
    c20 = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    c21 = A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]
    c22 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    inv = np.empty((3, 3))
    r = 1.0 / det
    # adjugate is the TRANSPOSE of the cofactor matrix
    inv[0, 0] = c00 * r
    inv[0, 1] = c10 * r
    inv[0, 2] = c20 * r
    inv[1, 0] = c01 * r
    inv[1, 1] = c11 * r
    inv[1, 2] = c21 * r
    inv[2, 0] = c02 * r
    inv[2, 1] = c12 * r
    inv[2, 2] = c22 * r
    return inv @ B, True


@njit(cache=True)
def kalman_forward_backward_ref(
    ts, ns, xs, var_c, var_m, lambda_,
    m_p, P_p, m_f, P_f, m_s, P_s,
    out_ms, out_vs,
):
    """Readable reference implementation of the Kalman/RTS recursion.

    Superseded in the hot path by `kalman_forward_backward`, which is the same
    recursion written out in scalars. Kept because it is the legible statement
    of the algorithm and serves as the oracle for the equivalence test in
    tests/test_kickscore.py — the scalar version is fast but unreadable, and
    hand-expanded 3x3 algebra is exactly the kind of code that grows a silent
    sign error (as the EP site update once did).
    """
    n = len(ts)
    if n == 0:
        return

    h = _H
    I3 = np.eye(3)

    # Initialise first time step
    m_p[0] = np.zeros(3)
    P_p[0] = compute_initial_cov(var_c, var_m, lambda_)

    # Forward pass
    for i in range(n):
        if i > 0:
            dt = ts[i] - ts[i - 1]
            A = compute_transition(dt, lambda_)
            Q = compute_noise_cov(dt, var_m, lambda_)
            m_p[i] = A @ m_f[i - 1]
            P_p[i] = A @ P_f[i - 1] @ A.T + Q

        # Kalman update with pseudo-observation
        Ph = P_p[i] @ h
        denom = 1.0 + xs[i] * (h @ Ph)
        k = Ph / denom

        m_f[i] = m_p[i] + k * (ns[i] - xs[i] * (h @ m_p[i]))

        # Joseph form for numerical stability
        Z = I3 - xs[i] * np.outer(k, h)
        P_f[i] = Z @ P_p[i] @ Z.T + xs[i] * np.outer(k, k)

    # Backward pass (RTS smoother)
    # No jitter: an absolute 1e-10 is a ~1e-5 RELATIVE perturbation of the f'
    # diagonal (var_m * lambda^2 ~ 9e-6 at lscale=365), and P_p's condition
    # number here is only ~1e4 — it bought nothing and biased the result.
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            m_s[i] = m_f[i].copy()
            P_s[i] = P_f[i].copy()
        else:
            dt = ts[i + 1] - ts[i]
            A = compute_transition(dt, lambda_)
            # G = (A @ P_f[i])' @ P_p[i+1]^{-1}
            # Solve P_p[i+1] @ G' = A @ P_f[i]
            sol, ok = _solve3(P_p[i + 1], A @ P_f[i])
            if not ok:
                # Singular predicted covariance — leave this step unsmoothed
                # rather than propagate NaNs backward through the chain.
                m_s[i] = m_f[i].copy()
                P_s[i] = P_f[i].copy()
                out_ms[i] = h @ m_s[i]
                out_vs[i] = h @ P_s[i] @ h
                continue
            G = sol.T
            m_s[i] = m_f[i] + G @ (m_s[i + 1] - m_p[i + 1])
            P_s[i] = P_f[i] + G @ (P_s[i + 1] - P_p[i + 1]) @ G.T

        out_ms[i] = h @ m_s[i]
        out_vs[i] = h @ P_s[i] @ h


@njit(cache=True)
def kalman_forward_backward(
    ts, ns, xs, var_c, var_m, lambda_,
    m_p, P_p, m_f, P_f, m_s, P_s,
    out_ms, out_vs,
):
    """Kalman forward pass + RTS backward pass for one player.

    Identical recursion to `kalman_forward_backward_ref`, written in scalars.
    The reference spends ~5.4us per appearance on a handful of 3x3 flops
    because every `@`, `np.outer` and kernel builder allocates a temporary
    array inside numba; this runs the same maths in registers. Since EP calls
    it for every player on every iteration, it dominated the whole sweep.

    Exploits the structure: the state is [c, f, f'], the constant component
    does not mix with the Matern block under the transition (A[0,0]=1, no
    off-diagonal, no process noise on c), the measurement is h=[1,1,0], and
    every covariance is symmetric so only 6 entries are tracked.

    Parameters
    ----------
    ts : float64 array (n_obs,) — observation times
    ns : float64 array (n_obs,) — EP pseudo-obs natural means
    xs : float64 array (n_obs,) — EP pseudo-obs precisions
    var_c, var_m, lambda_ : kernel parameters
    m_p, P_p, m_f, P_f, m_s, P_s : (n_obs, 3) / (n_obs, 3, 3) work arrays
    out_ms, out_vs : (n_obs,) output marginal means and variances
    """
    n = len(ts)
    if n == 0:
        return
    a = lambda_

    # ── forward pass ──────────────────────────────────────────────────
    for i in range(n):
        if i == 0:
            mp0 = 0.0
            mp1 = 0.0
            mp2 = 0.0
            pp00 = var_c
            pp01 = 0.0
            pp02 = 0.0
            pp11 = var_m
            pp12 = 0.0
            pp22 = var_m * a * a
        else:
            dt = ts[i] - ts[i - 1]
            adt = a * dt
            e = math.exp(-adt)
            a11 = e * (adt + 1.0)
            a12 = e * dt
            a21 = e * (-a * a * dt)
            a22 = e * (1.0 - adt)
            cc = math.exp(-2.0 * adt)
            q11 = var_m * (1.0 - cc * (2.0 * adt * adt + 2.0 * adt + 1.0))
            q12 = var_m * cc * (2.0 * adt * adt * a)
            q22 = var_m * a * a * (1.0 - cc * (2.0 * adt * adt - 2.0 * adt + 1.0))

            f0 = m_f[i - 1, 0]
            f1 = m_f[i - 1, 1]
            f2 = m_f[i - 1, 2]
            mp0 = f0
            mp1 = a11 * f1 + a12 * f2
            mp2 = a21 * f1 + a22 * f2

            p00 = P_f[i - 1, 0, 0]
            p01 = P_f[i - 1, 0, 1]
            p02 = P_f[i - 1, 0, 2]
            p11 = P_f[i - 1, 1, 1]
            p12 = P_f[i - 1, 1, 2]
            p22 = P_f[i - 1, 2, 2]
            # B = A @ P_f   (row 0 of A is [1,0,0], so B[0,:] = P_f[0,:])
            b01 = p01
            b02 = p02
            b11 = a11 * p11 + a12 * p12
            b12 = a11 * p12 + a12 * p22
            b21 = a21 * p11 + a22 * p12
            b22 = a21 * p12 + a22 * p22
            # P_p = B @ A.T + Q
            pp00 = p00
            pp01 = a11 * b01 + a12 * b02
            pp02 = a21 * b01 + a22 * b02
            pp11 = a11 * b11 + a12 * b12 + q11
            pp12 = a21 * b11 + a22 * b12 + q12
            pp22 = a21 * b21 + a22 * b22 + q22

        m_p[i, 0] = mp0
        m_p[i, 1] = mp1
        m_p[i, 2] = mp2
        P_p[i, 0, 0] = pp00
        P_p[i, 0, 1] = pp01
        P_p[i, 1, 0] = pp01
        P_p[i, 0, 2] = pp02
        P_p[i, 2, 0] = pp02
        P_p[i, 1, 1] = pp11
        P_p[i, 1, 2] = pp12
        P_p[i, 2, 1] = pp12
        P_p[i, 2, 2] = pp22

        # Kalman update with the EP pseudo-observation. h = [1,1,0], so
        # P_p @ h is just the sum of the first two columns.
        x = xs[i]
        ph0 = pp00 + pp01
        ph1 = pp01 + pp11
        ph2 = pp02 + pp12
        denom = 1.0 + x * (ph0 + ph1)
        k0 = ph0 / denom
        k1 = ph1 / denom
        k2 = ph2 / denom

        innov = ns[i] - x * (mp0 + mp1)
        mf0 = mp0 + k0 * innov
        mf1 = mp1 + k1 * innov
        mf2 = mp2 + k2 * innov
        m_f[i, 0] = mf0
        m_f[i, 1] = mf1
        m_f[i, 2] = mf2

        # Joseph form: Z = I - x * outer(k, h); with h = [1,1,0] the third
        # column of outer(k,h) is zero, so Z[:,2] = [0,0,1].
        z00 = 1.0 - x * k0
        z01 = -x * k0
        z10 = -x * k1
        z11 = 1.0 - x * k1
        z20 = -x * k2
        z21 = -x * k2
        # W = Z @ P_p  (Z[r,2] is 0 for r<2, and 1 for r=2)
        w00 = z00 * pp00 + z01 * pp01
        w01 = z00 * pp01 + z01 * pp11
        w02 = z00 * pp02 + z01 * pp12
        w10 = z10 * pp00 + z11 * pp01
        w11 = z10 * pp01 + z11 * pp11
        w12 = z10 * pp02 + z11 * pp12
        w20 = z20 * pp00 + z21 * pp01 + pp02
        w21 = z20 * pp01 + z21 * pp11 + pp12
        w22 = z20 * pp02 + z21 * pp12 + pp22
        # P_f = W @ Z.T + x * outer(k, k)
        pf00 = w00 * z00 + w01 * z01 + x * k0 * k0
        pf01 = w00 * z10 + w01 * z11 + x * k0 * k1
        pf02 = w00 * z20 + w01 * z21 + w02 + x * k0 * k2
        pf11 = w10 * z10 + w11 * z11 + x * k1 * k1
        pf12 = w10 * z20 + w11 * z21 + w12 + x * k1 * k2
        pf22 = w20 * z20 + w21 * z21 + w22 + x * k2 * k2
        P_f[i, 0, 0] = pf00
        P_f[i, 0, 1] = pf01
        P_f[i, 1, 0] = pf01
        P_f[i, 0, 2] = pf02
        P_f[i, 2, 0] = pf02
        P_f[i, 1, 1] = pf11
        P_f[i, 1, 2] = pf12
        P_f[i, 2, 1] = pf12
        P_f[i, 2, 2] = pf22

    # ── backward pass (RTS smoother) ──────────────────────────────────
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            for r in range(3):
                m_s[i, r] = m_f[i, r]
                for c in range(3):
                    P_s[i, r, c] = P_f[i, r, c]
        else:
            dt = ts[i + 1] - ts[i]
            adt = a * dt
            e = math.exp(-adt)
            a11 = e * (adt + 1.0)
            a12 = e * dt
            a21 = e * (-a * a * dt)
            a22 = e * (1.0 - adt)

            # C = A @ P_f[i]   (row 0 of A is [1,0,0])
            p00 = P_f[i, 0, 0]
            p01 = P_f[i, 0, 1]
            p02 = P_f[i, 0, 2]
            p11 = P_f[i, 1, 1]
            p12 = P_f[i, 1, 2]
            p22 = P_f[i, 2, 2]
            c00 = p00
            c01 = p01
            c02 = p02
            c10 = a11 * p01 + a12 * p02
            c11 = a11 * p11 + a12 * p12
            c12 = a11 * p12 + a12 * p22
            c20 = a21 * p01 + a22 * p02
            c21 = a21 * p11 + a22 * p12
            c22 = a21 * p12 + a22 * p22

            # inv(P_p[i+1]) via cofactors
            n00 = P_p[i + 1, 0, 0]
            n01 = P_p[i + 1, 0, 1]
            n02 = P_p[i + 1, 0, 2]
            n11 = P_p[i + 1, 1, 1]
            n12 = P_p[i + 1, 1, 2]
            n22 = P_p[i + 1, 2, 2]
            d00 = n11 * n22 - n12 * n12
            d01 = n12 * n02 - n01 * n22
            d02 = n01 * n12 - n11 * n02
            det = n00 * d00 + n01 * d01 + n02 * d02
            if abs(det) < 1e-300:
                # Singular predicted covariance — leave this step unsmoothed
                # rather than propagate NaNs backward through the chain.
                for r in range(3):
                    m_s[i, r] = m_f[i, r]
                    for c in range(3):
                        P_s[i, r, c] = P_f[i, r, c]
                out_ms[i] = m_s[i, 0] + m_s[i, 1]
                out_vs[i] = (P_s[i, 0, 0] + 2.0 * P_s[i, 0, 1] + P_s[i, 1, 1])
                continue
            r_ = 1.0 / det
            d11 = n00 * n22 - n02 * n02
            d12 = n02 * n01 - n00 * n12
            d22 = n00 * n11 - n01 * n01
            # inverse is symmetric (P_p symmetric)
            i00 = d00 * r_
            i01 = d01 * r_
            i02 = d02 * r_
            i11 = d11 * r_
            i12 = d12 * r_
            i22 = d22 * r_

            # sol = inv(P_p[i+1]) @ C ;  G = sol.T
            s00 = i00 * c00 + i01 * c10 + i02 * c20
            s01 = i00 * c01 + i01 * c11 + i02 * c21
            s02 = i00 * c02 + i01 * c12 + i02 * c22
            s10 = i01 * c00 + i11 * c10 + i12 * c20
            s11 = i01 * c01 + i11 * c11 + i12 * c21
            s12 = i01 * c02 + i11 * c12 + i12 * c22
            s20 = i02 * c00 + i12 * c10 + i22 * c20
            s21 = i02 * c01 + i12 * c11 + i22 * c21
            s22 = i02 * c02 + i12 * c12 + i22 * c22
            g00 = s00
            g01 = s10
            g02 = s20
            g10 = s01
            g11 = s11
            g12 = s21
            g20 = s02
            g21 = s12
            g22 = s22

            # m_s[i] = m_f[i] + G @ (m_s[i+1] - m_p[i+1])
            dm0 = m_s[i + 1, 0] - m_p[i + 1, 0]
            dm1 = m_s[i + 1, 1] - m_p[i + 1, 1]
            dm2 = m_s[i + 1, 2] - m_p[i + 1, 2]
            m_s[i, 0] = m_f[i, 0] + g00 * dm0 + g01 * dm1 + g02 * dm2
            m_s[i, 1] = m_f[i, 1] + g10 * dm0 + g11 * dm1 + g12 * dm2
            m_s[i, 2] = m_f[i, 2] + g20 * dm0 + g21 * dm1 + g22 * dm2

            # P_s[i] = P_f[i] + G @ (P_s[i+1] - P_p[i+1]) @ G.T
            e00 = P_s[i + 1, 0, 0] - P_p[i + 1, 0, 0]
            e01 = P_s[i + 1, 0, 1] - P_p[i + 1, 0, 1]
            e02 = P_s[i + 1, 0, 2] - P_p[i + 1, 0, 2]
            e11 = P_s[i + 1, 1, 1] - P_p[i + 1, 1, 1]
            e12 = P_s[i + 1, 1, 2] - P_p[i + 1, 1, 2]
            e22 = P_s[i + 1, 2, 2] - P_p[i + 1, 2, 2]
            # T = G @ E
            t00 = g00 * e00 + g01 * e01 + g02 * e02
            t01 = g00 * e01 + g01 * e11 + g02 * e12
            t02 = g00 * e02 + g01 * e12 + g02 * e22
            t10 = g10 * e00 + g11 * e01 + g12 * e02
            t11 = g10 * e01 + g11 * e11 + g12 * e12
            t12 = g10 * e02 + g11 * e12 + g12 * e22
            t20 = g20 * e00 + g21 * e01 + g22 * e02
            t21 = g20 * e01 + g21 * e11 + g22 * e12
            t22 = g20 * e02 + g21 * e12 + g22 * e22
            # P_s = P_f + T @ G.T
            ps00 = P_f[i, 0, 0] + t00 * g00 + t01 * g01 + t02 * g02
            ps01 = P_f[i, 0, 1] + t00 * g10 + t01 * g11 + t02 * g12
            ps02 = P_f[i, 0, 2] + t00 * g20 + t01 * g21 + t02 * g22
            ps11 = P_f[i, 1, 1] + t10 * g10 + t11 * g11 + t12 * g12
            ps12 = P_f[i, 1, 2] + t10 * g20 + t11 * g21 + t12 * g22
            ps22 = P_f[i, 2, 2] + t20 * g20 + t21 * g21 + t22 * g22
            P_s[i, 0, 0] = ps00
            P_s[i, 0, 1] = ps01
            P_s[i, 1, 0] = ps01
            P_s[i, 0, 2] = ps02
            P_s[i, 2, 0] = ps02
            P_s[i, 1, 1] = ps11
            P_s[i, 1, 2] = ps12
            P_s[i, 2, 1] = ps12
            P_s[i, 2, 2] = ps22

        # out = h @ m_s, h @ P_s @ h  with h = [1,1,0]
        out_ms[i] = m_s[i, 0] + m_s[i, 1]
        out_vs[i] = P_s[i, 0, 0] + 2.0 * P_s[i, 0, 1] + P_s[i, 1, 1]


@njit(cache=True)
def predict_at_time(t, ts, m_f, P_f, m_s, P_s, m_p_arr, P_p_arr,
                    var_c, var_m, lambda_):
    """Predict mean and variance at an arbitrary time t for one player."""
    n = len(ts)
    h = _H

    if n == 0:
        P0 = compute_initial_cov(var_c, var_m, lambda_)
        return 0.0, float(h @ P0 @ h)

    if t >= ts[-1]:
        # After last observation — propagate forward from smoothed
        dt = t - ts[-1]
        A = compute_transition(dt, lambda_)
        Q = compute_noise_cov(dt, var_m, lambda_)
        m = A @ m_s[-1]
        P = A @ P_s[-1] @ A.T + Q
        return float(h @ m), float(h @ P @ h)

    # Find position via binary search
    idx = np.searchsorted(ts, t)

    if idx == 0:
        # Before first observation
        m0 = np.zeros(3)
        P0 = compute_initial_cov(var_c, var_m, lambda_)
        return float(h @ m0), float(h @ P0 @ h)

    # Between observations idx-1 and idx
    j = idx - 1
    dt1 = t - ts[j]
    A1 = compute_transition(dt1, lambda_)
    Q1 = compute_noise_cov(dt1, var_m, lambda_)
    P = A1 @ P_f[j] @ A1.T + Q1
    m = A1 @ m_f[j]

    # RTS correction using right neighbour
    dt2 = ts[idx] - t
    A2 = compute_transition(dt2, lambda_)
    sol, ok = _solve3(P_p_arr[idx], A2 @ P)
    if not ok:
        # Singular — fall back to the uncorrected forward prediction.
        return float(h @ m), float(h @ P @ h)
    G = sol.T
    m_pred = m + G @ (m_s[idx] - m_p_arr[idx])
    P_pred = P + G @ (P_s[idx] - P_p_arr[idx]) @ G.T

    return float(h @ m_pred), float(h @ P_pred @ h)


# ── EP update for a single match ─────────────────────────────────────


@njit(cache=True)
def mm_win(mean_cav, cov_cav, obs_type):
    """Moment matching for a win, dispatched by likelihood.

    obs_type: 0 = logit (Bradley-Terry), 1 = probit (Thurstone).
    An int flag rather than a callable, so numba can still cache-compile.
    """
    if obs_type == 1:
        return _mm_probit_win(mean_cav, cov_cav)
    return mm_logit_win(mean_cav, cov_cav)


@njit(cache=True)
def _ep_site(coeff, dlogpart, d2logpart, n_cav, x_cav):
    """Natural parameters of one player's EP site.

    `coeff` is +1 for the winner and -1 for the loser. Keep it symbolic: it
    appears SQUARED on the `denom` and `x` terms and on the second term of `n`,
    but to the first power on the leading `dlogpart`. Hand-expanding the two
    cases is how the loser's site previously acquired a sign error.

    Returns (n, x, ok); ok is False if the site is degenerate.
    """
    cc = coeff * coeff
    denom = 1.0 + cc * d2logpart / x_cav
    if abs(denom) < 1e-12:
        return 0.0, 0.0, False
    x = -cc * d2logpart / denom
    n = coeff * (dlogpart - coeff * (n_cav / x_cav) * d2logpart) / denom
    return n, x, True


@njit(cache=True)
def ep_update_match(
    p1_ms, p1_vs, p1_ns, p1_xs, p1_idx,
    p2_ms, p2_vs, p2_ns, p2_xs, p2_idx,
    lr, obs_type,
):
    """EP update for one match (player 1 won).

    Modifies p1_ns[p1_idx], p1_xs[p1_idx], p2_ns[p2_idx], p2_xs[p2_idx]
    in place.

    Returns the absolute change in log-partition for convergence checking.
    """
    # Marginal posterior mean and variance for f = s1 - s2
    m1 = p1_ms[p1_idx]
    v1 = p1_vs[p1_idx]
    m2 = p2_ms[p2_idx]
    v2 = p2_vs[p2_idx]

    # Guard against zero/negative variance (numerical collapse)
    if v1 <= 1e-12 or v2 <= 1e-12:
        return 0.0

    # Cavity for player 1's contribution
    x1_tot = 1.0 / v1
    n1_tot = x1_tot * m1
    x1_cav = x1_tot - p1_xs[p1_idx]
    n1_cav = n1_tot - p1_ns[p1_idx]

    # Cavity for player 2's contribution
    x2_tot = 1.0 / v2
    n2_tot = x2_tot * m2
    x2_cav = x2_tot - p2_xs[p2_idx]
    n2_cav = n2_tot - p2_ns[p2_idx]

    # Function-space cavity: f = (+1)*s1 + (-1)*s2
    # Coefficients: c1 = +1, c2 = -1
    if x1_cav <= 0.0 or x2_cav <= 0.0:
        return 0.0  # skip degenerate cavities

    f_mean_cav = n1_cav / x1_cav - n2_cav / x2_cav
    f_var_cav = 1.0 / x1_cav + 1.0 / x2_cav

    # Moment matching (player 1 wins)
    logpart, dlogpart, d2logpart = mm_win(f_mean_cav, f_var_cav, obs_type)

    # Both sites are computed before either is committed: a mid-update bail-out
    # would otherwise leave the match half-applied, with player 1 updated
    # against a player 2 that never moved.
    n1_new, x1_new, ok1 = _ep_site(1.0, dlogpart, d2logpart, n1_cav, x1_cav)
    n2_new, x2_new, ok2 = _ep_site(-1.0, dlogpart, d2logpart, n2_cav, x2_cav)
    if not (ok1 and ok2):
        return 0.0

    p1_ns[p1_idx] = (1.0 - lr) * p1_ns[p1_idx] + lr * n1_new
    p1_xs[p1_idx] = (1.0 - lr) * p1_xs[p1_idx] + lr * x1_new
    p2_ns[p2_idx] = (1.0 - lr) * p2_ns[p2_idx] + lr * n2_new
    p2_xs[p2_idx] = (1.0 - lr) * p2_xs[p2_idx] + lr * x2_new

    return abs(logpart)
