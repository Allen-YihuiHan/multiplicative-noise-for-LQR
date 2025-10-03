import numpy as np
import matplotlib.pyplot as plt
import time
from utils import LQRSystem, RidgeAccumulator, synthesize_lqr_controller, collect_random_data, \
    make_random_lqr, mb_controller_from_estimate, infinite_horizon_cost, cost_safe_update, \
    batch_pg, dim_aware_hyperparams, clip_grad_fro, _sanitize

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
    state_grid = [5, 10]
    input_grid = [4, 8, 16, 32, 64, 128, 256]

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
            fname = f"../plots/multip/optimality_gap/state{n}_input{m}.png"
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