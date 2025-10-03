# Implements Multiple-trajectory Averaging Least-Squares (MALS)

import numpy as np
from numpy.linalg import lstsq, pinv
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
from utils import (                              # <-- reuse your existing helpers
    LQRSystem,
    collect_random_data,                         # (not used by MALS, but harmless to keep)
    make_random_lqr,
    mb_controller_from_estimate,                 # optional safety gate (DARE)
    infinite_horizon_cost, cost_safe_update,     # unchanged MF flow
    batch_pg, dim_aware_hyperparams, clip_grad_fro, _sanitize
)

# -----------------------------------------------------------------------------
# Utilities for symmetric vectorization (matches paper’s P1,Q1,P2,Q2)  [Eq. (5)]
# -----------------------------------------------------------------------------

def _vec_f(M):
    """Column-major vec compatible with A⊗B conventions."""
    return M.reshape(-1, order="F")

def build_elim_dup(n):
    """
    Build elimination (P) and duplication (Q) matrices so that:
        svec(X) = P vec(X),      vec(X) = Q svec(X)
    svec keeps the upper-triangular (i<=j) of X in column-major order.
    """
    r = n * (n + 1) // 2
    P = np.zeros((r, n * n))
    Q = np.zeros((n * n, r))
    k = 0
    for j in range(n):            # column-major
        for i in range(n):
            if i <= j:
                col = i + j * n
                P[k, col] = 1.0
                Q[col, k] = 1.0
                if i != j:
                    col_sym = j + i * n
                    Q[col_sym, k] = 1.0
                k += 1
    return P, Q

def svec(X, P):          # X is (n x n), P is from build_elim_dup(n)
    return P @ _vec_f(X)

# -----------------------------------------------------------------------------
# MALS core (Algorithm 1 in the paper): input design, data collection, LS fits
#   - First moments:   μ_{t+1} = A μ_t + B ν_t                         [Eq. (3)]
#   - Second moments:  X̃_{t+1} = (Ã+Σ̃A') X̃_t + (B̃+Σ̃B') Ũ_t
#                      + K_BA Ŵ_t + K_AB Ŵ'_t                         [Eq. (5)]
# -----------------------------------------------------------------------------

def design_inputs(m, ell, rng=None, mean_scale=1.0, cov_scale=1.0):
    """
    Simple input design for Algorithm 1 (Steps 1–3):
      - ν_t ~ N(0, mean_scale^2 I)
      - Ū_t = cov_scale^2 * I   (PSD)
    You can replace this with periodic or Wishart designs if desired.
    """
    rng = np.random.default_rng() if rng is None else rng
    nus  = np.stack([rng.standard_normal(m) * mean_scale for _ in range(ell)], axis=0)  # (ell, m)
    Ubar = np.stack([ (cov_scale**2) * np.eye(m)           for _ in range(ell)], axis=0)  # (ell, m, m)
    return nus, Ubar

def collect_rollouts_mals(env, ell, nr, nus, Ubar, x0_sampler=None, rng=None):
    """
    Algorithm 1 (Steps 4–10): collect nr rollouts of length ell under
    u_t^(k) ~ N(ν_t, Ū_t), with independently sampled x_0^(k).
    """
    rng = np.random.default_rng() if rng is None else rng
    n, m = env.n_states, env.n_inputs

    # storage
    X = np.zeros((nr, ell + 1, n))
    U = np.zeros((nr, ell, m))

    for k in range(nr):
        # step 5: initial state
        x0 = rng.standard_normal(n) if x0_sampler is None else np.asarray(x0_sampler(), float).reshape(n)
        env.reset()
        env.x = x0.reshape(n, 1)  # use your class as-is
        X[k, 0, :] = x0

        for t in range(ell):
            # steps 6–8: sample inputs with the designed first/second moments
            u = rng.multivariate_normal(mean=nus[t], cov=Ubar[t])
            U[k, t, :] = u
            x_next = env.step(u.reshape(m, 1))
            X[k, t+1, :] = x_next.flatten()

    return X, U  # shapes (nr, ell+1, n), (nr, ell, m)

def mals_fit(A_dim, B_dim, X, U, nus, Ubar):
    """
    Algorithm 1 (Steps 11–16).
    Inputs:
        - X: (nr, ell+1, n) states
        - U: (nr, ell,   m) inputs
        - nus:  (ell, m)  first moments   ν_t
        - Ubar: (ell, m, m) second central moments  Ū_t (PSD)

    Returns:
        A_hat, B_hat, SigmaA_tilde_hat, SigmaB_tilde_hat
    """
    nr, T1, n = X.shape
    ell = T1 - 1
    m = U.shape[2]

    # ----- build P1,Q1,P2,Q2 and shorthand sizes -----
    P1, Q1 = build_elim_dup(n)
    P2, Q2 = build_elim_dup(m)
    r_x = P1.shape[0]        # n(n+1)/2
    r_u = P2.shape[0]        # m(m+1)/2

    # ---------- Step 12: empirical moments over rollouts ----------
    # μ̂_t
    mu_hat = X[:, :ell+1, :].mean(axis=0)          # (ell+1, n)

    # X̃̂_t = (1/nr) P1 vec(Σ x_t x_t^T)
    Xtil_hat = np.zeros((ell+1, r_x))
    for t in range(ell+1):
        S = np.zeros((n, n))
        for k in range(nr):
            x = X[k, t, :].reshape(n, 1)
            S += x @ x.T
        S /= nr
        Xtil_hat[t, :] = svec(S, P1)

    # Ŵ_t = vec(μ̂_t ν_t^T),   Ŵ'_t = vec(ν_t μ̂_t^T)
    What  = np.zeros((ell, n * m))
    WpHat = np.zeros((ell, n * m))
    for t in range(ell):
        mu_t = mu_hat[t, :].reshape(n, 1)
        nu_t = nus[t].reshape(m, 1)
        What[t, :]  = _vec_f(mu_t @ nu_t.T)       # vec(x u^T)
        WpHat[t, :] = _vec_f(nu_t @ mu_t.T)       # vec(u x^T)

    # Ũ_t = P2 vec(Ū_t + ν_t ν_t^T)
    Util = np.zeros((ell, r_u))
    for t in range(ell):
        U2 = Ubar[t] + np.outer(nus[t], nus[t])
        Util[t, :] = svec(U2, P2)

    # ---------- Step 14: LS for [A B] with first moments ----------
    # Stack Y = [μ̂_1 ... μ̂_ell],  Z = [[μ̂_0 ... μ̂_{ell-1}]
    #                                     [ν_0  ... ν_{ell-1}]]
    Y = mu_hat[1:].T                                # (n, ell)
    Z = np.vstack([mu_hat[:-1].T, nus.T])          # (n+m, ell)
    Theta = Y @ Z.T @ pinv(Z @ Z.T)                # (n, n+m)
    A_hat = Theta[:, :n]
    B_hat = Theta[:, n:]

    # ---------- Step 15: build Ã, B̃, K_BA, K_AB from A_hat,B_hat ----------
    Atil = P1 @ np.kron(A_hat, A_hat) @ Q1         # (r_x, r_x)
    Btil = P1 @ np.kron(B_hat, B_hat) @ Q2         # (r_x, r_u)
    K_BA = P1 @ np.kron(B_hat, A_hat)              # (r_x, n*m)
    K_AB = P1 @ np.kron(A_hat, B_hat)              # (r_x, n*m)

    # ---------- Step 16: LS for [Σ̃A'  Σ̃B'] ----------
    # For each t:
    #   y_t = X̃̂_{t+1} - [Ã X̃̂_t + K_BA Ŵ_t + K_AB Ŵ'_t + B̃ Ũ_t]
    #   y_t = Σ̃A' X̃̂_t + Σ̃B' Ũ_t
    Y2 = np.zeros((r_x, ell))
    Z2 = np.zeros((r_x + r_u, ell))
    for t in range(ell):
        rhs_det = (
            Atil @ Xtil_hat[t] +
            K_BA @ What[t]     +
            K_AB @ WpHat[t]    +
            Btil @ Util[t]
        )
        resid = Xtil_hat[t+1] - rhs_det
        Y2[:, t] = resid
        Z2[:, t] = np.concatenate([Xtil_hat[t], Util[t]], axis=0)

    M = Y2 @ Z2.T @ pinv(Z2 @ Z2.T)                # (r_x, r_x + r_u)
    SigmaA_tilde_hat = M[:, :r_x]
    SigmaB_tilde_hat = M[:, r_x:]

    return A_hat, B_hat, SigmaA_tilde_hat, SigmaB_tilde_hat


def delta_sampler(t):
    # e.g., heteroscedastic Student-t or a regime-switching process
    return 2 * np.random.standard_t(df=5, size=8)

def gamma_sampler(t):
    # e.g., time-of-day cycles
    # base = np.array([0.02, 0.01, 0.04])
    # cycle = 0.5 + 0.5 * np.sin(2 * np.pi * t / 50.0)
    # return base * np.random.randn(3) * (1.0 + cycle)
    return 2 * np.random.standard_t(df=5, size=6)


def run():
    # sizes (you can adjust)
    state_grid = [5, 10]
    input_grid = [4, 8, 16, 32, 64, 128, 256]

    results = {}

    for n in state_grid:
        for m in input_grid:
            if m > n:
                continue

            # ----- build a nominal multiplicative plant, exactly as you already do -----
            # You likely already have a factory that returns (A, B, Ais, Bjs, alphas, betas).
            # Reuse it; here we just create a simple one for reference:
            rng = np.random.default_rng(42)

            A_TRUE, B_TRUE, Ais, Bjs, alphas, betas= make_random_lqr(n, m, p=16, q=12)

            # A = np.diag(0.95 * (0.6 + 0.4 * rng.random(n)))
            # A += (0.05 / np.sqrt(n)) * rng.standard_normal((n, n))
            # # rescale to spectral radius 0.95
            # s = np.max(np.abs(np.linalg.eigvals(A)));  A *= 0.95 / (s + 1e-12)
            # B = rng.standard_normal((n, m))
            # U, S, Vt = np.linalg.svd(B, full_matrices=False)
            # S = np.maximum(S, 0.2)
            # B = (U * S) @ Vt

            # # small multiplicative-noise directions (unused by MALS formulas; only for env)
            # p, q = 4, 3
            # Ais = [0.03 * rng.standard_normal((n, n)) for _ in range(p)]
            # Bjs = [0.03 * rng.standard_normal((n, m)) for _ in range(q)]
            # alphas = (0.03**2) * np.ones(p)
            # betas  = (0.03**2) * np.ones(q)

            Q = np.eye(n)
            R = 0.1 * np.eye(m)
            NOISE_STD = 0.05

            # two envs with the same plant (model-free / model-based comparison)
            # env_mf = LQRSystem(A_TRUE, B_TRUE, Q, R, Ais=Ais, Bjs=Bjs, alphas=alphas, betas=betas, noise_std=NOISE_STD)
            # env_mb = LQRSystem(A_TRUE, B_TRUE, Q, R, Ais=Ais, Bjs=Bjs, alphas=alphas, betas=betas, noise_std=NOISE_STD)
            env_mf = LQRSystem(A_TRUE, B_TRUE, Q, R, Ais=Ais, Bjs=Bjs, 
                               alphas=alphas, betas=betas, noise_std=NOISE_STD,
                               delta_sampler=delta_sampler, gamma_sampler=gamma_sampler)
            env_mb = LQRSystem(A_TRUE, B_TRUE, Q, R, Ais=Ais, Bjs=Bjs,
                               alphas=alphas, betas=betas, noise_std=NOISE_STD,
                               delta_sampler=delta_sampler, gamma_sampler=gamma_sampler)


            # nominal LQR "oracle" for gap
            P_opt = solve_discrete_are(A_TRUE, B_TRUE, Q, R)
            J_opt = float(np.trace(P_opt))

            # MF hyperparams (same as your flow)
            PROJ_RADIUS, MAX_GRAD_NORM, _, _ = dim_aware_hyperparams(n, m)
            SIGMA0 = 0.5 / np.sqrt(m)
            LR0 = 5e-3 / np.sqrt(n*m)

            NUM_UPDATES    = 200
            HORIZON_LENGTH = 20
            N_ROLLOUTS_MF  = 16        # for REINFORCE batch
            NR_MALS        = 12        # number of rollouts for MALS
            RIDGE_LAMBDA   = 1e-3      # unused here; kept for symmetry

            # gates for mb_controller_from_estimate (optional safety)
            RHO_THR, ZETA_THR, PSI_THR, GAMMA_THR, EPS_R = 0.99, 50.0, 50.0, 1e-3, 1e-6

            K_pg = np.zeros((m, n))

            cum_steps_mf = cum_steps_mb = 0
            gaps_mf, gaps_mb = [], []
            steps_mf, steps_mb = [], []

            # cum_time_mf = cum_time_mb = 0.0
            # times_mf, times_mb = [], []
            rng = np.random.default_rng(123)

            for i in range(NUM_UPDATES):
                # ---------------- model-free step (unchanged) ----------------
                # t_start = time.perf_counter()
                SIGMA = max(SIGMA0 * (0.95**(i//10)), 0.1 * SIGMA0)
                grad, _ = batch_pg(env_mf, K_pg, HORIZON_LENGTH, SIGMA, N_ROLLOUTS_MF)
                grad = clip_grad_fro(grad, MAX_GRAD_NORM)
                LR = LR0 / np.sqrt(1 + i / 50)
                K_pg, _ = cost_safe_update(A_TRUE, B_TRUE, Q, R, K_pg, LR * grad, proj_radius=PROJ_RADIUS, margin=1e-2)

                # cum_time_mf += (time.perf_counter() - t_start)
                # times_mf.append(cum_time_mf)

                cum_steps_mf += HORIZON_LENGTH
                J_mf = infinite_horizon_cost(A_TRUE, B_TRUE, Q, R, K_pg)
                steps_mf.append(cum_steps_mf)
                gaps_mf.append(J_mf - J_opt)

                # ---------------- model-based step via MALS ------------------
                # Input design for this MALS batch

                # t_start = time.perf_counter()
                nus, Ubar = design_inputs(m, HORIZON_LENGTH, rng=rng, mean_scale=1.0, cov_scale=1.0)

                # Collect multiple trajectories as in Alg. 1 (reset per rollout)
                X, U = collect_rollouts_mals(env_mb, HORIZON_LENGTH, NR_MALS, nus, Ubar, x0_sampler=None, rng=rng)

                # Fit [A_hat, B_hat] and [Σ̃A', Σ̃B'] (the latter also returned for completeness)
                A_hat, B_hat, SigmaA_tilde_hat, SigmaB_tilde_hat = mals_fit(n, m, X, U, nus, Ubar)

                # Synthesize controller from nominal estimate (same as your baseline)
                K_mb, _ = mb_controller_from_estimate(A_hat, B_hat, RHO_THR, ZETA_THR, PSI_THR, GAMMA_THR, eps_R=EPS_R)

                # cum_time_mb += (time.perf_counter() - t_start)
                # times_mb.append(cum_time_mb)

                cum_steps_mb += HORIZON_LENGTH
                J_mb = infinite_horizon_cost(A_TRUE, B_TRUE, Q, R, K_mb)
                steps_mb.append(cum_steps_mb)
                gaps_mb.append(J_mb - J_opt)

            # ----- store (same shapes/names you already use) -----
            results[(n, m)] = {
                "steps_mf": np.array(steps_mf, dtype=int),
                "gaps_mf":  np.array(gaps_mf, dtype=float),
                "steps_mb": np.array(steps_mb, dtype=int),
                "gaps_mb":  np.array(gaps_mb, dtype=float),
                # "time_mf":  np.array(times_mf, dtype=float),
                # "time_mb":  np.array(times_mb, dtype=float),
                "J_opt": J_opt,
            }

            print(f"[n={n}, m={m}] final gaps: MF={gaps_mf[-1]:.3g}  MB(MALS+DARE)={gaps_mb[-1]:.3g}")


            # ---------- per-pair plot ----------
            g_mf = _sanitize(results[(n,m)]["gaps_mf"])
            g_mb = _sanitize(results[(n,m)]["gaps_mb"])

            plt.figure(figsize=(6,4))
            plt.plot(results[(n,m)]["steps_mf"], g_mf, label="Model-free (REINFORCE)")
            plt.plot(results[(n,m)]["steps_mb"], g_mb, label="Model-based (MALS+DARE)")
            plt.xlabel("# samples (time steps)")
            plt.ylabel("Optimality gap")
            plt.title(f"Optimality Gap vs Samples  (states={n}, input={m})")
            plt.grid(True)
            plt.yscale("log")
            plt.legend()
            fname = f"../plots/mals/optimality_gap/state{n}_input{m}.png"
            plt.tight_layout()
            plt.savefig(fname, dpi=180)
            plt.close()
            print(f"[n={n}, m={m}] saved {fname}")

            # ---------- per-pair runtime plot ----------
            # plt.figure(figsize=(6,4))
            # updates = np.arange(1, len(results[(n,m)]["time_mf"]) + 1)
            # plt.plot(updates, results[(n,m)]["time_mf"], label="Model-free time (s)")
            # plt.plot(updates, results[(n,m)]["time_mb"], label="Model-based time (s)")
            # plt.xlabel("Update round")
            # plt.ylabel("Cumulative wall-clock time (s)")
            # plt.title(f"Runtime vs Updates  (states={n}, input={m})")
            # plt.grid(True)
            # plt.legend()
            # fname_rt = f"runtime_state{n}_input{m}.png"
            # plt.tight_layout()
            # plt.savefig(fname_rt, dpi=150)
            # plt.close()
            # print(f"[n={n}, m={m}] saved {fname_rt}")

    return results


if __name__ == "__main__":
    run()
