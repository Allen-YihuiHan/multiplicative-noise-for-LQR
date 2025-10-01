import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov


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
    G = np.zeros_like(K); ret = 0.0
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