import time
import numpy as np


def sinkhorn(a, b, C, epsilon, max_iter=1000, max_time=20, delta=1e-8, lbd=1e-10):
    """Classic sinkhorn algorithm."""
    start = time.time()

    # Get kernel and initial value
    K = np.exp(-C / epsilon)
    v = np.ones(b.shape[0])

    # Store data
    err_p = []
    err_q = []
    times = []
    cost = []

    n_iter = 0
    err = 1 + delta
    time_actual = time.time() - start
    while (n_iter < max_iter) and (time_actual < max_time) and (err > delta):
        n_iter += 1

        # Sinkhorn step 1
        u = a / (K @ v + lbd)

        # Error computation
        r = v * (K.T @ u + lbd)
        err_q_curr = np.linalg.norm(r - b, 1)
        err_q.append(err_q_curr)

        # Sinkhorn step 2
        v = b / (K.T @ u + lbd)

        # Error computation
        s = u * (K @ v + lbd)
        err_p_curr = np.linalg.norm(s - a, 1)
        err_p.append(err_p_curr)

        # Update error
        err = max(err_p_curr, err_q_curr)

        # Get cost
        P = (np.diag(u) @ K) @ np.diag(v)
        cost_curr = np.trace(C.T @ P)
        cost.append(cost_curr)

        # Update time
        time_actual = time.time() - start
        times.append(time_actual)

    return P, np.array(err_p), np.array(err_q), np.array(times), np.array(cost)


def lr_ot_trivial_init(a, b, rank):
    """Trivial initialisation values for Q, R and g."""
    g = np.ones(rank) / rank
    Q = a[:, None] @ g[:, None].T
    R = b[:, None] @ g[:, None].T
    return Q, R, g


def lr_ot_random_init(a, b, rank, lbd=1e-2):
    """Gaussian initialisation values for Q, R and g."""
    n, m = np.shape(a)[0], np.shape(b)[0]

    g = np.abs(np.random.randn(rank)) + lbd
    g = g / np.sum(g)

    Q = np.abs(np.random.randn(n, rank)) + lbd
    Q = (Q.T * a / np.sum(Q, axis=1)).T

    R = np.abs(np.random.randn(m, rank)) + lbd
    R = (R.T * b / np.sum(R, axis=1)).T

    return Q, R, g


def lr_dykstra(
    xi1,
    xi2,
    xi3,
    p1,
    p2,
    alpha=1e-7,
    max_iter=1000,
    delta=1e-9,
    lbd=0
):
    """Algorithm 2 of Low-Rank Sinkhorn Factorisation."""
    err = []

    # In case max_iter==0
    Q = xi1
    R = xi2
    g_tilde = xi3

    # Initialization
    n, m = np.shape(p1)[0], np.shape(p2)[0]
    r = np.shape(xi3)[0]

    v1_tilde, v2_tilde = np.ones(r), np.ones(r)
    u1, u2 = np.ones(n), np.ones(m)

    q13, q23 = np.ones(r), np.ones(r)
    q1, q2 = np.ones(r), np.ones(r)

    err_curr = 1 + delta
    n_iter = 0
    while n_iter < max_iter and err_curr > delta:

        # Store old values in case of overflow
        u1_prev, v1_prev = u1, v1_tilde
        u2_prev, v2_prev = u2, v2_tilde
        g_prev = g_tilde

        # Update number of iterations
        n_iter = n_iter + 1

        # Update values
        u1 = p1 / (xi1 @ v1_tilde + lbd)
        u2 = p2 / (xi2 @ v2_tilde + lbd)

        g = np.maximum(alpha, g_tilde * q13)
        q13 = (g_tilde * q13) / (g + lbd)

        g_tilde = g.copy()

        g = np.power(g_tilde * q23, 1/3)
        g *= np.power(v1_tilde * q1 * (xi1.T @ u1), 1 / 3)
        g *= np.power(v2_tilde * q2 * (xi2.T @ u2), 1 / 3)

        v1 = g / ((xi1.T @ u1) + lbd)
        v2 = g / ((xi2.T @ u2) + lbd)

        q1 = (q1 * v1_tilde) / (v1 + lbd)
        q2 = (q2 * v2_tilde) / (v2 + lbd)
        q23 = (g_tilde * q23) / (g + lbd)

        # Copy values to avoid errors
        v1_tilde = v1.copy()
        v2_tilde = v2.copy()
        g_tilde = g.copy()

        # Update the error
        err_curr = np.linalg.norm(u1 * (xi1 @ v1) - p1, 1)
        err_curr += np.linalg.norm(u2 * (xi2 @ v2) - p2, 1)
        err.append(err_curr)

        # Be sure that all the values are finite
        if not (np.any(np.isfinite(u1))
                and np.any(np.isfinite(u2))
                and np.any(np.isfinite(v1))
                and np.any(np.isfinite(v2))
        ):
            u1, v1 = u1_prev, v1_prev
            u2, v2 = u2_prev, v2_prev
            g = g_prev
            break

    Q = np.diag(u1) @ xi1 @ np.diag(v1)
    R = np.diag(u2) @ xi2 @ np.diag(v2)
    return Q, R, g, np.array(err)


def KL(X, Y, lbd=1e-10):
    """Kulback-Leibler divergence."""
    return np.sum(X * (np.log(X + lbd) - np.log(Y + lbd)))


def low_rank_ot(
    a,
    b,
    rank,
    C,
    epsilon=0,
    alpha=1e-10,
    max_iter=1000,
    delta=1e-3,
    max_time=100,
    max_iter_dykstra=1000,
    delta_dykstra=1e-3,
    lbd_dykstra=0,
    lbd_kl=0,
    gamma=None,
    lbd_init=1
):
    """Low rank Optimal Transport solver."""
    start = time.time()

    cost = []
    err = []
    times = []

    # Initial values
    Q, R, g = lr_ot_random_init(a, b, rank, lbd=lbd_init)

    # Rescale cost matrix
    C /= np.max(C)

    # Get step
    if gamma is None:
        L = (2 / (alpha ** 4)) * (np.linalg.norm(C) ** 2)
        L += ((epsilon + 2 * np.linalg.norm(C)) / (alpha ** 3)) ** 2
        L = np.sqrt(3 * L)
        gamma = 1 / L

    # Get time
    time_actual = time.time() - start

    err_curr = 1 + delta
    n_iter = 0
    count_escape = 1
    while (n_iter < max_iter) and (time_actual < max_time) and (err_curr > delta):

        # Store old values
        Q_prev = Q.copy()
        R_prev = R.copy()
        g_prev = g.copy()

        # Update iteration
        n_iter = n_iter + 1

        # Update values
        omega = np.diag(Q.T @ C @ R)

        xi1 = (C @ R) / g
        xi1 += epsilon * np.log(Q)
        xi1 -= (1 / gamma) * np.log(Q)
        xi1 = np.exp((-gamma) * xi1)

        xi2 = (C.T @ Q) / g
        xi2 += epsilon * np.log(R)
        xi2 -= (1 / gamma) * np.log(R)
        xi2 = np.exp((-gamma) * xi2)

        xi3 = -omega / (g ** 2)
        xi3 += epsilon * np.log(g)
        xi3 -= (1 / gamma) * np.log(g)
        xi3 = np.exp((-gamma) * xi3)

        Q, R, g, _ = lr_dykstra(
            xi1,
            xi2,
            xi3,
            a,
            b,
            alpha=alpha,
            max_iter=max_iter_dykstra,
            delta=delta_dykstra,
            lbd=lbd_dykstra,
        )

        # Compute cost and error
        cost_curr = C @ R / g
        cost_curr = Q.T @ cost_curr
        cost_curr = np.trace(cost_curr)

        criterion = (KL(Q, Q_prev, lbd=lbd_kl) + KL(Q_prev, Q, lbd=lbd_kl))
        criterion += (KL(R, R_prev, lbd=lbd_kl) + KL(R_prev, R, lbd=lbd_kl))
        criterion += (KL(g, g_prev, lbd=lbd_kl) + KL(g_prev, g, lbd=lbd_kl))
        criterion *= ((1 / gamma) ** 2)

        # Check that the cost and the error are well-defined
        if np.isnan(criterion) or np.isnan(cost_curr):
            if np.isnan(criterion):
                print("Break: criterion is nan at iter ", n_iter)
            else:
                print("Break: cost is nan at iter:", n_iter)
            Q = Q_prev
            R = R_prev
            g = g_prev
            break

        err.append(err_curr)
        cost.append(cost_curr)

        # Avoid early stop
        if n_iter > 1:
            # If the criterion is way larger than delta
            if criterion > delta * 10:
                err_curr = criterion

            # If the criterion is comparable or smaller with delta
            else:
                count_escape += 1
                if count_escape != n_iter:
                    err_curr = criterion

        time_actual = time.time() - start
        times.append(time_actual)

    return np.array(err), np.array(cost), np.array(times), Q, R, g


def low_rank_ot_linear(
    a,
    b,
    rank,
    C,
    C1,
    C2,
    epsilon=0,
    alpha=1e-10,
    max_iter=1000,
    delta=1e-3,
    max_time=100,
    max_iter_dykstra=10000,
    delta_dykstra=1e-3,
    lbd_dykstra=0,
    lbd_kl=0,
    gamma=None,
    lbd_init=1
):
    """Linear version of the Low Rank Optimal Transport solver."""
    start = time.time()
    cost = []
    err = []
    times = []

    # Initial values
    Q, R, g = lr_ot_random_init(a, b, rank, lbd=lbd_init)

    # Rescale cost matrices
    C1 /= np.sqrt(np.max(C1))
    C2 /= np.sqrt(np.max(C2))

    # Compute sub-cost once
    CR = C2.T @ R
    CQ = C1.T @ Q

    # Get step
    if gamma is None:
        C = C1 @ C2.T
        L = (2 / (alpha ** 4)) * (np.linalg.norm(C) ** 2)
        L += ((epsilon + 2 * np.linalg.norm(C)) / (alpha ** 3)) ** 2
        L = np.sqrt(3 * L)
        gamma = 1 / L

    # Get time
    time_actual = time.time() - start

    err_curr = 1 + delta
    n_iter = 0
    count_escape = 1
    while (n_iter < max_iter) and (time_actual < max_time) and (err_curr > delta):

        # Store old values
        Q_prev = Q
        R_prev = R
        g_prev = g

        # Update iteration
        n_iter = n_iter + 1

        # Update values
        xi1 = C1 @ CR / g
        xi1 += epsilon * np.log(Q)
        xi1 -= (1 / gamma) * np.log(Q)
        xi1 = np.exp((-gamma) * xi1)

        xi2 = C2 @ CQ / g
        xi2 += epsilon * np.log(R)
        xi2 -= (1 / gamma) * np.log(R)
        xi2 = np.exp((-gamma) * xi2)

        omega = np.diag(CQ.T @ CR)

        xi3 = -omega / (g**2)
        xi3 += epsilon * np.log(g)
        xi3 -= (1 / gamma) * np.log(g)
        xi3 = np.exp((-gamma) * xi3)

        Q, R, g, _ = lr_dykstra(
            xi1,
            xi2,
            xi3,
            a,
            b,
            alpha=alpha,
            max_iter=max_iter_dykstra,
            delta=delta_dykstra,
            lbd=lbd_dykstra
        )

        # Update sub-cost
        CR = C2.T @ R
        CQ = C1.T @ Q

        # Compute cost and error
        cost_curr = np.trace(CQ.T @ CR / g)

        criterion = (KL(Q, Q_prev, lbd=lbd_kl) + KL(Q_prev, Q, lbd=lbd_kl))
        criterion += (KL(R, R_prev, lbd=lbd_kl) + KL(R_prev, R, lbd=lbd_kl))
        criterion += (KL(g, g_prev, lbd=lbd_kl) + KL(g_prev, g, lbd=lbd_kl))
        criterion *= ((1 / gamma) ** 2)

        # Check that the cost and the error are well-defined
        if np.isnan(criterion) or np.isnan(cost_curr):
            if np.isnan(criterion):
                print("Break: criterion is nan at iter ", n_iter)
            else:
                print("Break cost is nan at iter ", n_iter)
            Q = Q_prev
            R = R_prev
            g = g_prev
            break

        err.append(err_curr)
        cost.append(cost_curr)

        # Avoid early stop
        if n_iter > 1:
            # If the criterion is way larger than delta
            # We can safely update the error
            if criterion > delta * 10:
                err_curr = criterion

            # If the criterion is comparable or smaller with delta
            # We update the number of possible escape
            else:
                count_escape += 1
                if count_escape != n_iter:
                    err_curr = criterion

        time_actual = time.time() - start
        times.append(time_actual)

    return np.array(err), np.array(cost), np.array(times), Q, R, g
