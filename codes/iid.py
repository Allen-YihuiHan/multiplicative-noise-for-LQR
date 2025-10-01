import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov
import matplotlib.pyplot as plt
import time


#   LQR Environment
class LQRSystem:
    def __init__(self, A_true, B_true, Q, R, noise_std=0.1):
        self.A_true = A_true
        self.B_true = B_true
        self.Q = Q
        self.R = R
        self.noise_std = noise_std
        self.n_states = A_true.shape[0]
        self.n_inputs = B_true.shape[1]
        self.reset()

    def reset(self):
        self.x = np.zeros((self.n_states, 1))
        return self.x

    def step(self, u):
        """
        x_{t+1} = A*x_t + B*u_t + w_t
        """
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        w = self.noise_std * np.random.randn(self.n_states, 1)
        self.x = self.A_true @ self.x + self.B_true @ u + w
        return self.x


class RidgeAccumulator:
    def __init__(self, n, m):
        self.p = n + m
        self.n = n
        self.G = np.zeros((self.p, self.p))
        self.H = np.zeros((self.p, n))

    def update(self, X, X_next, U):
        # X, X_next: (T, n); U: (T, m)  (rows are time samples)
        Z = np.hstack([X, U])             # (T, n+m)
        self.G += Z.T @ Z
        self.H += Z.T @ X_next

    def solve(self, lam):
        theta = np.linalg.solve(self.G + lam * np.eye(self.p), self.H)  # (n+m, n)
        n = self.n
        A_hat = theta[:n, :].T
        B_hat = theta[n:, :].T
        return A_hat, B_hat


def synthesize_lqr_controller(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    # K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return P


#   Data collection
def collect_random_data(env, horizon, act_std):
    """
    Returns arrays stacked over time: X, X_next, U.
    """
    X, Xn, U = [], [], []
    x = env.reset()
    for _ in range(horizon):
        u = act_std * np.random.randn(env.n_inputs, 1)
        X.append(x.flatten())
        U.append(u.flatten())
        x = env.step(u)
        Xn.append(x.flatten())
    return np.array(X), np.array(Xn), np.array(U)



def mb_controller_from_estimate(Ah, Bh, varrho, zeta, psi, gamma, eps_R=1e-6):
    """
    Implements Alg. 3 lines 4–8:
      - safety checks on (Â,  B̂)
      - if pass, solve DARE with Q=I, R≈0, then K = (B^T P B)^(-1) B^T P A
      - returns (K, ok_flag)
    """
    # line 4: gates
    rhoA  = np.max(np.abs(np.linalg.eigvals(Ah)))
    normA = np.linalg.norm(Ah, 2)
    normB = np.linalg.norm(Bh, 2)
    sminB = np.linalg.svd(Bh, compute_uv=False)[-1]

    if (rhoA > varrho) or (normA > zeta) or (normB > psi) or (sminB < gamma):
        return np.zeros((Bh.shape[1], Ah.shape[0])), False  # K_plug = 0

    # lines 7–8: DARE with Q = I, R ≈ 0 (εI for numerics)
    n, m = Ah.shape[0], Bh.shape[1]
    try:
        P = solve_discrete_are(Ah, Bh, np.eye(n), eps_R * np.eye(m))
        # K here is the positive matrix; your simulator applies u = -Kx
        K = np.linalg.pinv(Bh.T @ P @ Bh) @ (Bh.T @ P @ Ah)
        return K, True
    except Exception:
        # if stabilizability fails numerically, fall back to zero controller
        return np.zeros((m, n)), False


#   Policy gradient
def collect_trajectory_pg(env, K, horizon, exploration_std):
    """
    Collects one trajectory using u = -Kx + eta, eta ~ N(0, sigma^2 I).
    """
    states, actions, rewards, exploration_noises = [], [], [], []
    x = env.reset()
    for _ in range(horizon):
        eta = exploration_std * np.random.randn(env.n_inputs, 1)
        u = -K @ x + eta
        cost = x.T @ env.Q @ x + u.T @ env.R @ u
        rewards.append(-cost.item())
        states.append(x.flatten())
        actions.append(u.flatten())
        exploration_noises.append(eta.flatten())
        x = env.step(u)
    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'exploration_noises': np.array(exploration_noises)
    }


def compute_policy_gradient(trajectory, exploration_std):
    states = trajectory['states']
    rewards = trajectory['rewards']
    exploration_noises = trajectory['exploration_noises']
    T = len(rewards)

    # returns-to-go as Ψ_t, normalized (variance reduction)
    G = 0.0
    returns = np.zeros(T)
    for t in reversed(range(T)):
        G = rewards[t] + G
        returns[t] = G
    eps = np.finfo(np.float32).eps.item()
    returns = (returns - np.mean(returns)) / (np.std(returns) + eps)

    grad = np.zeros((exploration_noises.shape[1], states.shape[1]))  # (m, n)
    for t in range(T):
        eta_t = exploration_noises[t, np.newaxis].T  # (m,1)
        x_t   = states[t, np.newaxis]                # (1,n)
        grad += -(eta_t @ x_t) * returns[t]
    return grad / (exploration_std ** 2)


def fro_norm(M):
    return np.sqrt(np.sum(M * M))

def clip_grad_fro(grad, max_norm):
    nrm = fro_norm(grad)
    if nrm > max_norm and nrm > 0.0:
        grad = grad * (max_norm / nrm)
    return grad

def project_fro(K, radius):
    nrm = fro_norm(K)
    if nrm > radius and nrm > 0.0:
        K = K * (radius / nrm)
    return K

def rho_cl(A, B, K):  # ρ(A - BK)
    return np.max(np.abs(np.linalg.eigvals(A - B @ K)))

def cost_safe_update(A, B, Q, R, K, step, proj_radius=None, margin=1e-2, shrink=0.5, max_tries=25):
    J_prev = infinite_horizon_cost(A, B, Q, R, K)
    K_new  = K + step
    if proj_radius is not None:
        K_new = project_fro(K_new, proj_radius)

    tries = 0
    while tries < max_tries:
        rho = rho_cl(A, B, K_new)
        if rho < 1.0 - margin:
            J_new = infinite_horizon_cost(A, B, Q, R, K_new)
            if np.isfinite(J_new) and J_new <= J_prev:
                return K_new, True
        step *= shrink
        K_new = K + step
        if proj_radius is not None:
            K_new = project_fro(K_new, proj_radius)
        tries += 1
    return K, False


def batch_pg(env, K, horizon, sigma, n_rollouts):
    G = np.zeros_like(K); ret = 0.0
    for _ in range(n_rollouts):
        traj = collect_trajectory_pg(env, K, horizon, sigma)
        G += compute_policy_gradient(traj, sigma)
        ret += traj['rewards'].sum()
    return G / n_rollouts, ret / n_rollouts


#   Cost utilities
def infinite_horizon_cost(A, B, Q, R, K):
    """
    J(K) = trace(P_K) with P_K solving:
      P_K = Q + K^T R K + (A-BK)^T P_K (A-BK)
    (Assumes Sigma_0 = I; proportional for any PSD Sigma_0.)
    """
    Acl = A - B @ K
    if np.max(np.abs(np.linalg.eigvals(Acl))) >= 1.0:
        return np.inf
    Qbar = Q + K.T @ R @ K
    Pk = solve_discrete_lyapunov(Acl.T, Qbar)
    return np.trace(Pk)


def make_random_lqr(n, m, rho_target=0.95, coupling=0.05, gamma_min=0.2, seed=None):
    """
    Returns (A,B) with spectral radius(A) ≈ rho_target and σ_min(B) ≥ gamma_min.
    - n: #states, m: #inputs
    - coupling controls off-diagonal strength (0 -> diagonal A)
    """
    rng = np.random.default_rng(seed)

    # Start from a stable diagonal A, then add small coupling and rescale to rho_target
    diag = rho_target * (0.6 + 0.4 * rng.random(n))      # eigenvalues in (0.57, 0.95)
    A = np.diag(diag)
    if coupling > 0:
        A += (coupling / np.sqrt(n)) * rng.standard_normal((n, n))
        s = np.max(np.abs(np.linalg.eigvals(A)))
        A *= (rho_target / (s + 1e-12))                  # enforce ρ(A) = rho_target

    # Random B then lift small singular values to guarantee σ_min(B) ≥ gamma_min
    B = rng.standard_normal((n, m))
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    S = np.maximum(S, gamma_min)
    B = (U * S) @ Vt
    return A, B

def dim_aware_hyperparams(n, m):
    # Heuristics that scale gently with dimension
    proj_radius   = 2.0 * np.sqrt(n * m)     # Frobenius radius for K
    max_grad_norm = 10.0 * np.sqrt(n * m)    # clip on REINFORCE gradient
    lr            = 1e-3 / np.sqrt(n * m)    # smaller lr for larger problems
    sigma         = 0.5  / np.sqrt(max(m,1)) # exploration std
    return proj_radius, max_grad_norm, lr, sigma


# Transpose infinity to nan
def _sanitize(y):
    y = np.asarray(y, dtype=float)
    y[~np.isfinite(y)] = np.nan
    return y


@profile
def name():
    # true system
    state_grid = [10]    # 500 is possible but slow; try later
    input_grid = [8, 16, 32, 64, 128, 256]

    # state_grid = [20, 50, 100]
    # input_grid = [16, 32]

    results= {}

    for n in state_grid:
        for m in input_grid:
            if m > n:      # skip non-sensical cases for basic tests
                continue

            # --- make system ---
            A_TRUE, B_TRUE = make_random_lqr(n, m, rho_target=0.95, coupling=0.05, gamma_min=0.2, seed=10_000 + 97*n + m)

            Q = np.eye(n)
            R = 0.1 * np.eye(m)
            NOISE_STD = 0.05

            # Envs
            env_mf = LQRSystem(A_TRUE, B_TRUE, Q, R, noise_std=NOISE_STD)
            env_mb = LQRSystem(A_TRUE, B_TRUE, Q, R, noise_std=NOISE_STD)

            # Ground-truth optimal
            P_opt = synthesize_lqr_controller(A_TRUE, B_TRUE, Q, R)
            J_opt = np.trace(P_opt)

            # Dim-aware PG params
            PROJ_RADIUS, MAX_GRAD_NORM, _, _ = dim_aware_hyperparams(n, m)
            SIGMA0 = 0.5 / np.sqrt(m)
            LR0 = 5e-3 / np.sqrt(n*m)

            # Identify & learn
            NUM_UPDATES    = 500
            HORIZON_LENGTH = 20
            MB_ACT_STD     = 1.0
            RIDGE_LAMBDA   = 1e-3

            # Gates for Alg. 3
            RHO_THR, ZETA_THR, PSI_THR, GAMMA_THR, EPS_R = 0.99, 50.0, 50.0, 1e-3, 1e-6

            # States
            K_pg = np.zeros((m, n))
            acc = RidgeAccumulator(n, m)
            cum_steps_mf = cum_steps_mb = 0
            gaps_mf, gaps_mb = [], []
            steps_mf, steps_mb = [], []
            # Timing trackers (cumulative per-method)
            cum_time_mf = 0.0
            cum_time_mb = 0.0
            times_mf, times_mb = [], []

            for i in range(NUM_UPDATES):
                # --- model-free step ---
                t_start = time.perf_counter()
                N_ROLLOUTS = 16
                SIGMA = max(SIGMA0 * (0.95**(i//10)), 0.1 * SIGMA0)
                grad, avg_ret = batch_pg(env_mf, K_pg, HORIZON_LENGTH, SIGMA, N_ROLLOUTS)
                grad = clip_grad_fro(grad, MAX_GRAD_NORM)
                # traj = collect_trajectory_pg(env_mf, K_pg, HORIZON_LENGTH, SIGMA)
                # grad = compute_policy_gradient(traj, SIGMA)
                # grad = clip_grad_fro(grad, MAX_GRAD_NORM)
                # K_pg = project_fro(K_pg + LR * grad, PROJ_RADIUS)
                LR = LR0 / np.sqrt(1 + i / 50)
                K_pg, _ = cost_safe_update(A_TRUE, B_TRUE, Q, R, K_pg, LR * grad, proj_radius=PROJ_RADIUS, margin=1e-2)

                # record time for model-free step (cumulative)
                cum_time_mf += (time.perf_counter() - t_start)
                times_mf.append(cum_time_mf)

                cum_steps_mf += HORIZON_LENGTH
                J_mf = infinite_horizon_cost(A_TRUE, B_TRUE, Q, R, K_pg)
                steps_mf.append(cum_steps_mf) 
                gaps_mf.append(J_mf - J_opt)

                # --- model-based step ---
                t_start = time.perf_counter()
                X, Xn, U = collect_random_data(env_mb, HORIZON_LENGTH, MB_ACT_STD)
                acc.update(X, Xn, U)
                A_hat, B_hat = acc.solve(RIDGE_LAMBDA)
                K_mb, _ = mb_controller_from_estimate(A_hat, B_hat, RHO_THR, ZETA_THR, PSI_THR, GAMMA_THR, eps_R=EPS_R)

                # record time for model-based step (cumulative)
                cum_time_mb += (time.perf_counter() - t_start)
                times_mb.append(cum_time_mb)

                cum_steps_mb += HORIZON_LENGTH
                J_mb = infinite_horizon_cost(A_TRUE, B_TRUE, Q, R, K_mb)
                steps_mb.append(cum_steps_mb)
                gaps_mb.append(J_mb - J_opt)
            
            print(f"[n={n}, m={m}]  final gaps  MF={gaps_mf[-1]:.3g}  MB={gaps_mb[-1]:.3g}")
            
            # ---------- store ----------
            results[(n, m)] = {
                "steps_mf": np.array(steps_mf, dtype=int),
                "gaps_mf":  np.array(gaps_mf, dtype=float),
                "steps_mb": np.array(steps_mb, dtype=int),
                "gaps_mb":  np.array(gaps_mb, dtype=float),
                "time_mf":  np.array(times_mf, dtype=float),
                "time_mb":  np.array(times_mb, dtype=float),
                "J_opt": float(J_opt),
            }

            # ---------- per-pair plot ----------
            g_mf = _sanitize(results[(n,m)]["gaps_mf"])
            g_mb = _sanitize(results[(n,m)]["gaps_mb"])

            plt.figure(figsize=(6,4))
            plt.plot(results[(n,m)]["steps_mf"], g_mf, label="Model-free (REINFORCE)")
            plt.plot(results[(n,m)]["steps_mb"], g_mb, label="Model-based (LS+DARE)")
            plt.xlabel("# samples (time steps)")
            plt.ylabel("Optimality gap")
            plt.title(f"Optimality Gap vs Samples  (states={n}, input={m})")
            plt.grid(True)
            plt.yscale("log")      # comment out if you prefer linear
            plt.legend()
            fname = f"state{n}_input{m}.png"
            plt.tight_layout()
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"[n={n}, m={m}] saved {fname}")

            # ---------- per-pair runtime plot (cumulative time per update) ----------
            plt.figure(figsize=(6,4))
            updates = np.arange(1, len(results[(n,m)]["time_mf"]) + 1)
            plt.plot(updates, results[(n,m)]["time_mf"], label="Model-free time (s)")
            plt.plot(updates, results[(n,m)]["time_mb"], label="Model-based time (s)")
            plt.xlabel("Update round")
            plt.ylabel("Cumulative wall-clock time (s)")
            plt.title(f"Runtime vs Updates  (states={n}, input={m})")
            plt.grid(True)
            plt.legend()
            fname_rt = f"runtime_state{n}_input{m}.png"
            plt.tight_layout()
            plt.savefig(fname_rt, dpi=150)
            plt.close()
            print(f"[n={n}, m={m}] saved {fname_rt}")

            
            # (Optionally: store results for plotting across sizes)


if __name__ == '__main__':

    name()