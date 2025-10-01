import numpy as np
import matplotlib.pyplot as plt
import time
from utils import RidgeAccumulator, synthesize_lqr_controller, collect_random_data, \
    mb_controller_from_estimate, infinite_horizon_cost, cost_safe_update, batch_pg, \
        dim_aware_hyperparams, clip_grad_fro, _sanitize


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


#   generates multiplicative structure
def make_random_lqr(
    n, m, p=4, q=3, a_scale=0.03, b_scale=0.03,
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
    state_grid = [5, 10, 20, 50]
    input_grid = [2, 4, 8, 16, 32, 64, 128, 256]

    results= {}

    for n in state_grid:
        for m in input_grid:
            if m > n:
                continue

            # --- make system (returns LQRm parts; no F(K) checks) ---
            # build tables: T steps × p (or q) indices
            # alpha_tbl = np.vstack([ (0.03**2) * (0.99**t) * np.ones(4) for t in range(2000)])  # example
            # beta_tbl  = np.vstack([ (0.03**2) * (0.95**t) * np.array([1,0.5,2]) for t in range(2000)])

            # A_TRUE, B_TRUE, Ais, Bjs, alphas, betas, alpha_sched, beta_sched = make_random_lqr(
            # n, m, p=4, q=3, return_schedules=True，
            # alpha_table=alpha_tbl, beta_table=beta_tbl)

            A_TRUE, B_TRUE, Ais, Bjs, alphas, betas= make_random_lqr(n, m, p=16, q=12)

            Q = np.eye(n)
            R = 0.1 * np.eye(m)
            NOISE_STD = 1

            # Environments: multiplicative-noise plant, learning unchanged
            env_mf = LQRSystem(A_TRUE, B_TRUE, Q, R, Ais=Ais, Bjs=Bjs, alphas=alphas, betas=betas, 
                               noise_std=NOISE_STD,
                               delta_sampler=delta_sampler, gamma_sampler=gamma_sampler)
            env_mb = LQRSystem(A_TRUE, B_TRUE, Q, R, Ais=Ais, Bjs=Bjs, alphas=alphas, betas=betas, 
                               noise_std=NOISE_STD,
                               delta_sampler=delta_sampler, gamma_sampler=gamma_sampler)

            # Ground-truth "optimal" for nominal model (same as before)
            P_opt = synthesize_lqr_controller(A_TRUE, B_TRUE, Q, R)
            J_opt = np.trace(P_opt)

            PROJ_RADIUS, MAX_GRAD_NORM, _, _ = dim_aware_hyperparams(n, m)
            SIGMA0 = 0.5 / np.sqrt(m)
            LR0 = 5e-3 / np.sqrt(n * m)

            NUM_UPDATES    = 500
            HORIZON_LENGTH = 20
            MB_ACT_STD     = 1.0
            RIDGE_LAMBDA   = 1e-3

            RHO_THR, ZETA_THR, PSI_THR, GAMMA_THR, EPS_R = 0.99, 50.0, 50.0, 1e-3, 1e-6

            K_pg = np.zeros((m, n))
            acc = RidgeAccumulator(n, m)

            cum_steps_mf = cum_steps_mb = 0
            gaps_mf, gaps_mb = [], []
            steps_mf, steps_mb = [], []

            # cum_time_mf = cum_time_mb = 0.0
            # times_mf, times_mb = [], []

            for i in range(NUM_UPDATES):
                # --- model-free step ---
                # t_start = time.perf_counter()
                N_ROLLOUTS = 16
                SIGMA = max(SIGMA0 * (0.95**(i//10)), 0.1 * SIGMA0)
                grad, avg_ret = batch_pg(env_mf, K_pg, HORIZON_LENGTH, SIGMA, N_ROLLOUTS)
                grad = clip_grad_fro(grad, MAX_GRAD_NORM)
                LR = LR0 / np.sqrt(1 + i / 50)
                K_pg, _ = cost_safe_update(A_TRUE, B_TRUE, Q, R, K_pg, LR * grad, proj_radius=PROJ_RADIUS, margin=1e-2)
                
                # cum_time_mf += (time.perf_counter() - t_start)
                # times_mf.append(cum_time_mf)

                cum_steps_mf += HORIZON_LENGTH
                J_mf = infinite_horizon_cost(A_TRUE, B_TRUE, Q, R, K_pg)
                steps_mf.append(cum_steps_mf)
                gaps_mf.append(J_mf - J_opt)

                # --- model-based step ---
                # t_start = time.perf_counter()
                X, Xn, U = collect_random_data(env_mb, HORIZON_LENGTH, MB_ACT_STD)
                acc.update(X, Xn, U)
                A_hat, B_hat = acc.solve(RIDGE_LAMBDA)
                K_mb, _ = mb_controller_from_estimate(A_hat, B_hat, RHO_THR, ZETA_THR, PSI_THR, GAMMA_THR, eps_R=EPS_R)
                
                # cum_time_mb += (time.perf_counter() - t_start)
                # times_mb.append(cum_time_mb)

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
                # "time_mf":  np.array(times_mf, dtype=float),
                # "time_mb":  np.array(times_mb, dtype=float),
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
            plt.yscale("log")
            plt.legend()
            fname = f"../plots/multiplicative/optimality_gap/state{n}_input{m}.png"
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


if __name__ == '__main__':
    run()