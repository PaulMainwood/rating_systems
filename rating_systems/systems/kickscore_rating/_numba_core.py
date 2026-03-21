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
def kalman_forward_backward(
    ts, ns, xs, var_c, var_m, lambda_,
    m_p, P_p, m_f, P_f, m_s, P_s,
    out_ms, out_vs,
):
    """Run Kalman forward pass + RTS backward pass for one player.

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
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            m_s[i] = m_f[i].copy()
            P_s[i] = P_f[i].copy()
        else:
            dt = ts[i + 1] - ts[i]
            A = compute_transition(dt, lambda_)
            # G = (A @ P_f[i])' @ P_p[i+1]^{-1}
            # Solve P_p[i+1] @ G' = A @ P_f[i]
            G = np.linalg.solve(P_p[i + 1], A @ P_f[i]).T
            m_s[i] = m_f[i] + G @ (m_s[i + 1] - m_p[i + 1])
            P_s[i] = P_f[i] + G @ (P_s[i + 1] - P_p[i + 1]) @ G.T

        out_ms[i] = h @ m_s[i]
        out_vs[i] = h @ P_s[i] @ h


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
    G = np.linalg.solve(P_p_arr[idx], A2 @ P).T
    m_pred = m + G @ (m_s[idx] - m_p_arr[idx])
    P_pred = P + G @ (P_s[idx] - P_p_arr[idx]) @ G.T

    return float(h @ m_pred), float(h @ P_pred @ h)


# ── EP update for a single match ─────────────────────────────────────


@njit(cache=True)
def ep_update_match(
    p1_ms, p1_vs, p1_ns, p1_xs, p1_idx,
    p2_ms, p2_vs, p2_ns, p2_xs, p2_idx,
    lr,
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

    # Moment matching (logistic likelihood, player 1 wins)
    logpart, dlogpart, d2logpart = mm_logit_win(f_mean_cav, f_var_cav)

    # Update player 1 (coeff = +1)
    denom1 = 1.0 + d2logpart / x1_cav
    if abs(denom1) < 1e-12:
        return 0.0
    x1_new = -d2logpart / denom1
    n1_new = (dlogpart - (n1_cav / x1_cav) * d2logpart) / denom1
    p1_ns[p1_idx] = (1.0 - lr) * p1_ns[p1_idx] + lr * n1_new
    p1_xs[p1_idx] = (1.0 - lr) * p1_xs[p1_idx] + lr * x1_new

    # Update player 2 (coeff = -1)
    denom2 = 1.0 + d2logpart / x2_cav
    if abs(denom2) < 1e-12:
        return 0.0
    x2_new = -d2logpart / denom2
    n2_new = (-dlogpart - (-n2_cav / x2_cav) * d2logpart) / denom2
    p2_ns[p2_idx] = (1.0 - lr) * p2_ns[p2_idx] + lr * n2_new
    p2_xs[p2_idx] = (1.0 - lr) * p2_xs[p2_idx] + lr * x2_new

    return abs(logpart)
