import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov


#   LQR Environment (multiplicative-noise plant)
class LQRSystem:
    def __init__(self, A_true, B_true, Q, R,
                 Ais=None, Bjs=None, alphas=None, betas=None, noise_std=0.0,
                 alpha_schedule=None, beta_schedule=None,
                 delta_sampler=None, gamma_sampler=None,
                 seed=None):
        """
        x_{t+1} = (A + Σ δ_{ti} A_i) x_t + (B + Σ γ_{tj} B_j) u_t + w_t
        δ_{ti} and γ_{tj} may vary with t and i/j.

        alpha_schedule(t) -> array of variances α_{t,·} (len p)
        beta_schedule(t)  -> array of variances β_{t,·} (len q)
        delta_sampler(t)  -> direct samples δ_{t,·} (len p)
        gamma_sampler(t)  -> direct samples γ_{t,·} (len q)
        """
        self.A_true, self.B_true, self.Q, self.R = A_true, B_true, Q, R
        self.noise_std = noise_std
        self.n_states = A_true.shape[0]
        self.n_inputs = B_true.shape[1]

        self.Ais = [] if Ais is None else Ais
        self.Bjs = [] if Bjs is None else Bjs
        self.alphas = np.zeros(len(self.Ais)) if alphas is None else np.asarray(alphas, float)
        self.betas  = np.zeros(len(self.Bjs)) if betas  is None else np.asarray(betas,  float)

        # new: optional schedules / samplers / rng
        self.alpha_schedule = alpha_schedule
        self.beta_schedule  = beta_schedule
        self.delta_sampler  = delta_sampler
        self.gamma_sampler  = gamma_sampler
        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        self.x = np.zeros((self.n_states, 1))
        self.t = 0  # track time for schedules/samplers
        return self.x

    def _draw_deltas(self):
        p = len(self.Ais)
        if p == 0: 
            return None
        if self.delta_sampler is not None:
            return np.asarray(self.delta_sampler(self.t), float)  # shape (p,)
        if self.alpha_schedule is not None:
            var = np.asarray(self.alpha_schedule(self.t), float)  # (p,)
            std = np.sqrt(var)
            return std * self.rng.standard_normal(p)
        # fallback: constant per-index variances (old behavior)
        return np.sqrt(self.alphas) * self.rng.standard_normal(p)

    def _draw_gammas(self):
        q = len(self.Bjs)
        if q == 0:
            return None
        if self.gamma_sampler is not None:
            return np.asarray(self.gamma_sampler(self.t), float)
        if self.beta_schedule is not None:
            var = np.asarray(self.beta_schedule(self.t), float)
            std = np.sqrt(var)
            return std * self.rng.standard_normal(q)
        return np.sqrt(self.betas) * self.rng.standard_normal(q)

    def _sample_Atilde_Btilde(self):
        deltas = self._draw_deltas()
        gammas = self._draw_gammas()

        Atilde = self.A_true if deltas is None else \
                 self.A_true + sum(d * Ai for d, Ai in zip(deltas, self.Ais))
        Btilde = self.B_true if gammas is None else \
                 self.B_true + sum(g * Bj for g, Bj in zip(gammas, self.Bjs))
        return Atilde, Btilde

    def step(self, u):
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        Atilde, Btilde = self._sample_Atilde_Btilde()
        w = self.noise_std * self.rng.standard_normal((self.n_states, 1))
        self.x = Atilde @ self.x + Btilde @ u + w
        self.t += 1
        return self.x


class RidgeAccumulator:
    def __init__(self, n, m):
        self.p = n + m
        self.n = n
        self.G = np.zeros((self.p, self.p))
        self.H = np.zeros((self.p, n))

    def update(self, X, X_next, U):
        Z = np.hstack([X, U])
        self.G += Z.T @ Z
        self.H += Z.T @ X_next

    def solve(self, lam):
        theta = np.linalg.solve(self.G + lam * np.eye(self.p), self.H)
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
    

#   generates multiplicative structure
def make_random_lqr(
    n, m, p=0, q=0, a_scale=0.03, b_scale=0.03,
    rho_target=0.95, coupling=0.05, gamma_min=0.2,
    alphas=None, betas=None,                  # per-index base variances (length p / q)
    alpha_table=None, beta_table=None,        # time×index variance tables -> schedules
    alpha_decay=None, beta_decay=None,        # scalars in (0,1): geometric decay schedules
    return_schedules=False, seed=None         # if True, also return the callables
):
    """
    Returns:
      A, B, {A_i}, {B_j}, {α_i}, {β_j}  [and optionally alpha_schedule, beta_schedule]

    If you want time-varying variances α_{t,i}, β_{t,j}:
      - Provide alpha_table with shape (T, p) and/or beta_table with shape (T, q); or
      - Provide alpha_decay / beta_decay (geometric: α_t = α_0 * decay^t); or
      - Later, pass your own sampler via LQRSystem(delta_sampler=..., gamma_sampler=...).
    """
    rng = np.random.default_rng(seed)

    # --- nominal A, B (same as before) ---
    diag = rho_target * (0.6 + 0.4 * rng.random(n))
    A = np.diag(diag)
    if coupling > 0:
        A += (coupling / np.sqrt(n)) * rng.standard_normal((n, n))
        s = np.max(np.abs(np.linalg.eigvals(A)))
        A *= (rho_target / (s + 1e-12))

    B = rng.standard_normal((n, m))
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    S = np.maximum(S, gamma_min)
    B = (U * S) @ Vt

    if p == 0 and q== 0:
        return A, B
    
    else:
        # --- multiplicative-noise directions ---
        Ais = [a_scale * rng.standard_normal((n, n)) for _ in range(p)]
        Bjs = [b_scale * rng.standard_normal((n, m)) for _ in range(q)]

        # base per-index variances (used if no schedule overrides)
        if alphas is None:
            alphas = (a_scale ** 2) * np.ones(p)
        else:
            alphas = np.asarray(alphas, float).reshape(p)

        if betas is None:
            betas = (b_scale ** 2) * np.ones(q)
        else:
            betas = np.asarray(betas, float).reshape(q)

        # ---- build optional schedules (as callables) ----
        alpha_schedule = None
        beta_schedule  = None

        def _wrap_table(table, dim):
            table = np.asarray(table, float)
            assert table.ndim == 2 and table.shape[1] == dim, "table must be (T, dim)"
            T = table.shape[0]
            def sched(t):
                idx = t if t < T else T - 1   # hold last row after T
                return table[idx]
            return sched

        if alpha_table is not None:
            alpha_schedule = _wrap_table(alpha_table, p)
        elif alpha_decay is not None:
            a0 = alphas.copy()
            d = float(alpha_decay)

            def alpha_schedule(t, a0=a0, d=d): 
                return a0 * (d ** t)

        if beta_table is not None:
            beta_schedule = _wrap_table(beta_table, q)
        elif beta_decay is not None:
            b0 = betas.copy()
            d = float(beta_decay)

            def beta_schedule(t, b0=b0, d=d): 
                return b0 * (d ** t)

        if return_schedules:
            return A, B, Ais, Bjs, alphas, betas, alpha_schedule, beta_schedule
        else:
            return A, B, Ais, Bjs, alphas, betas


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
    G = np.zeros_like(K)
    ret = 0.0

    for _ in range(n_rollouts):
        traj = collect_trajectory_pg(env, K, horizon, sigma)
        G += compute_policy_gradient(traj, sigma)
        ret += traj['rewards'].sum()

    return G / n_rollouts, ret / n_rollouts


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