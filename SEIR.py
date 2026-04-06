### Question 7.2 (a)

# load data
import jax.numpy as jnp
import jax
import pandas as pd
import numpy as np
import pypomp as pp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import pickle

# run level: 0 = quick debug, 1 = laptop, 2 = Great Lakes GPU
RL = 2

cache_dir = f"cache_{RL}"
os.makedirs(cache_dir, exist_ok=True)

file = "Measles_Consett_1948.csv"
meas = (pd.read_csv(file)
    .loc[:, ["week", "cases"]]
    .rename(columns={"week": "time", "cases": "reports"})
    .set_index("time")
    .astype(float))
ys = meas.copy()
ys.columns = pd.Index(["reports"])

def rinit(theta_, key, covars, t0):
    """Initial state simulator for SEIR model."""
    N = theta_["N"]
    eta = theta_["eta"]
    iota = theta_["iota"]   # initial fraction exposed
    S0 = jnp.round(N * eta)
    E0 = jnp.round(N * iota)
    I0 = 1.0
    R0 = N - S0 - E0 - I0
    H0 = 0.0
    return {"S": S0, "E": E0, "I": I0, "R": R0, "H": H0}

def rproc(X_, theta_, key, covars, t, dt):
    """Process simulator for SEIR model using Euler-binomial scheme."""
    S = jnp.asarray(X_["S"])
    E = jnp.asarray(X_["E"])
    I = jnp.asarray(X_["I"])
    R = jnp.asarray(X_["R"])
    H = jnp.asarray(X_["H"])

    Beta = theta_["Beta"]
    mu_EI = theta_["mu_EI"]
    mu_IR = theta_["mu_IR"]
    N = theta_["N"]

    p_SE = 1.0 - jnp.exp(-Beta * I / N * dt)
    p_EI = 1.0 - jnp.exp(-mu_EI * dt)
    p_IR = 1.0 - jnp.exp(-mu_IR * dt)

    key_SE, key_EI, key_IR = jax.random.split(key, 3)

    dN_SE = jax.random.binomial(key_SE, n=jnp.int32(S), p=p_SE)
    dN_EI = jax.random.binomial(key_EI, n=jnp.int32(E), p=p_EI)
    dN_IR = jax.random.binomial(key_IR, n=jnp.int32(I), p=p_IR)

    return {
        "S": S - dN_SE,
        "E": E + dN_SE - dN_EI,
        "I": I + dN_EI - dN_IR,
        "R": R + dN_IR,
        "H": H + dN_EI
    }

def nbinom_logpmf(x, k, mu):
    """Log PMF of NegBin(k, mu) robust when mu == 0."""
    x = jnp.asarray(x)
    k = jnp.asarray(k)
    mu = jnp.asarray(mu)

    logp_zero = jnp.where(x == 0, 0.0, -jnp.inf)
    safe_mu = jnp.where(mu == 0.0, 1.0, mu)

    core = (
        jax.scipy.special.gammaln(k + x)
        - jax.scipy.special.gammaln(k)
        - jax.scipy.special.gammaln(x + 1)
        + k * jnp.log(k / (k + safe_mu))
        + x * jnp.log(safe_mu / (k + safe_mu))
    )
    return jnp.where(mu == 0.0, logp_zero, core)

def rnbinom(key, k, mu):
    """Sample from NegBin(k, mu) via Gamma-Poisson mixture."""
    key_g, key_p = jax.random.split(key)
    lam = jax.random.gamma(key_g, k) * (mu / k)
    return jax.random.poisson(key_p, lam)

def dmeas(Y_, X_, theta_, covars, t):
    """Measurement density: log P(reports | H, rho, k)."""
    rho = theta_["rho"]
    k = theta_["k"]
    H = X_["H"]
    mu = rho * H
    return nbinom_logpmf(Y_["reports"], k, mu)

def rmeas(X_, theta_, key, covars, t):
    """Measurement simulator."""
    rho = theta_["rho"]
    k = theta_["k"]
    H = X_["H"]
    mu = rho * H
    reports = rnbinom(key, k, mu)
    return jnp.array([reports])

theta = {
    "Beta": 20.0,
    "mu_EI": 0.5,
    "mu_IR": 0.875,   # fixed
    "N": 38000.0,     # fixed
    "eta": 0.06,
    "iota": 0.0007,
    "rho": 0.8,
    "k": 20.0         # fixed
}

statenames = ["S", "E", "I", "R", "H"]
EST_VARS = ["Beta", "mu_EI", "eta", "iota", "rho"]

def build_seir_model(theta_input):
    return pp.Pomp(
        rinit=rinit,
        rproc=rproc,
        dmeas=dmeas,
        rmeas=rmeas,
        ys=ys,
        theta=theta_input,
        statenames=statenames,
        par_trans=par_trans_seir,
        t0=0.0,
        nstep=7,
        accumvars=("H",),
        ydim=1,
        covars=None
    )

def to_est_seir(theta):
    return {
        "Beta": jnp.log(theta["Beta"]),
        "mu_EI": jnp.log(theta["mu_EI"]),
        "mu_IR": jnp.log(theta["mu_IR"]),
        "N": theta["N"],
        "eta": jax.scipy.special.logit(theta["eta"]),
        "iota": jax.scipy.special.logit(theta["iota"]),
        "rho": jax.scipy.special.logit(theta["rho"]),
        "k": jnp.log(theta["k"]),
    }

def from_est_seir(theta):
    return {
        "Beta": jnp.exp(theta["Beta"]),
        "mu_EI": jnp.exp(theta["mu_EI"]),
        "mu_IR": jnp.exp(theta["mu_IR"]),
        "N": theta["N"],
        "eta": jax.scipy.special.expit(theta["eta"]),
        "iota": jax.scipy.special.expit(theta["iota"]),
        "rho": jax.scipy.special.expit(theta["rho"]),
        "k": jnp.exp(theta["k"]),
    }

par_trans_seir = pp.ParTrans(
    to_est=to_est_seir,
    from_est=from_est_seir
)

settings = {
    0: {
        "J_mif": 200,
        "M_local": 8,
        "M_global_1": 8,
        "M_global_2": 12,
        "n_local": 4,
        "n_starts": 12,
        "J_eval": 500,
        "eval_reps": 2,
    },
    1: {
    "J_mif": 250,
    "M_local": 10,
    "M_global_1": 10,
    "M_global_2": 14,
    "n_local": 5,
    "n_starts": 10,
    "J_eval": 600,
    "eval_reps": 2,
    },
    2: {
    "J_mif": 300,
    "M_local": 10,
    "M_global_1": 10,
    "M_global_2": 15,
    "n_local": 6,
    "n_starts": 16,
    "J_eval": 800,
    "eval_reps": 3,
    }
}

cfg = settings[RL]

rw_sd = pp.RWSigma(
    sigmas={
        "Beta": 0.02,
        "mu_EI": 0.02,
        "mu_IR": 0.0,   # fixed
        "N": 0.0,       # fixed
        "eta": 0.02,
        "iota": 0.02,
        "rho": 0.02,
        "k": 0.0        # fixed
    },
    init_names=["eta", "iota"]
)

rw_sd_small = pp.RWSigma(
    sigmas={
        "Beta": 0.01,
        "mu_EI": 0.01,
        "mu_IR": 0.0,
        "N": 0.0,
        "eta": 0.01,
        "iota": 0.01,
        "rho": 0.01,
        "k": 0.0
    },
    init_names=["eta", "iota"]
)

mod_test = build_seir_model(theta)
key = jax.random.key(42)
mod_test.pfilter(key=key, J=cfg["J_eval"], reps=1)
test_result = mod_test.results_history.last()
print(float(test_result.logLiks.values[0, 0]))

### Local Search

cache_file = f"{cache_dir}/seir_local_mif.pkl"

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        local_mif_result, local_mif_time = pickle.load(f)
else:
    n_local = cfg["n_local"]
    theta_list = [theta.copy() for _ in range(n_local)]
    mod_local = build_seir_model(theta_list)

    t0 = time.perf_counter()
    key = jax.random.key(20260405)
    mod_local.mif(
        J=cfg["J_mif"],
        M=cfg["M_local"],
        rw_sd=rw_sd,
        a=0.5,
        key=key
    )
    local_mif_time = time.perf_counter() - t0
    local_mif_result = mod_local.results_history.last()

    with open(cache_file, "wb") as f:
        pickle.dump((local_mif_result, local_mif_time), f)

print(f"Local mif time: {local_mif_time:.2f} sec")
print(
    "Time per IF2 iteration per 1000 particles:",
    local_mif_time / (cfg["M_local"] * (cfg["J_mif"] / 1000.0))
)

traces_da = local_mif_result.traces_da
n_reps = traces_da.sizes["replicate"]
iterations = traces_da.coords["iteration"].values

fig, axes = plt.subplots(3, 2, figsize=(8, 7))
plot_vars = ["logLik", "Beta", "mu_EI", "eta", "iota", "rho"]

for ax, var in zip(axes.flat, plot_vars):
    if var in traces_da.coords["variable"].values:
        for r in range(n_reps):
            ax.plot(
                iterations,
                traces_da.isel(replicate=r).sel(variable=var).values,
                alpha=0.4,
                lw=0.8
            )
    ax.set_title(var)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{cache_dir}/local_traces.png", dpi=200)
plt.close()

cache_file = f"{cache_dir}/seir_local_eval.pkl"

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        local_results, local_eval_time = pickle.load(f)
else:
    traces = local_mif_result.traces_da
    n_local = traces.sizes["replicate"]

    theta_endpoints = []
    for r in range(n_local):
        th = theta.copy()
        for var in EST_VARS:
            th[var] = float(
                traces.isel(replicate=r).sel(variable=var).values[-1]
            )
        theta_endpoints.append(th)

    mod_eval = build_seir_model(theta_endpoints)

    t0 = time.perf_counter()
    key = jax.random.key(20260406)
    mod_eval.pfilter(
        key=key,
        J=cfg["J_eval"],
        reps=cfg["eval_reps"]
    )
    local_eval_time = time.perf_counter() - t0
    pf = mod_eval.results_history.last()

    rows = []
    for i in range(n_local):
        lls = pf.logLiks.values[i, :]
        rows.append({
            **theta_endpoints[i],
            "loglik": pp.logmeanexp(lls),
            "loglik_se": pp.logmeanexp_se(lls)
        })

    local_results = pd.DataFrame(rows)
    local_results = local_results[np.isfinite(local_results["loglik"])]

    with open(cache_file, "wb") as f:
        pickle.dump((local_results, local_eval_time), f)

print(f"Local eval time: {local_eval_time:.2f} sec")
print(local_results.sort_values("loglik", ascending=False).head())

### Global Search

cache_file = f"{cache_dir}/seir_global_search.pkl"

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        global_results, global_mif_time, global_eval_time = pickle.load(f)
else:
    np.random.seed(20260407)
    n_starts = cfg["n_starts"]

    theta_list = []
    for _ in range(n_starts):
        th = theta.copy()
        th["Beta"] = np.random.uniform(5.0, 50.0)
        th["mu_EI"] = np.random.uniform(0.1, 2.0)
        th["eta"] = np.random.uniform(0.01, 0.20)
        th["iota"] = np.random.uniform(1e-5, 0.02)
        th["rho"] = np.random.uniform(0.2, 0.95)
        theta_list.append(th)

    mod_global = build_seir_model(theta_list)

    t0 = time.perf_counter()

    key = jax.random.key(20260408)
    mod_global.mif(
        J=cfg["J_mif"],
        M=cfg["M_global_1"],
        rw_sd=rw_sd,
        a=0.5,
        key=key
    )

    key = jax.random.key(20260409)
    mod_global.mif(
        J=cfg["J_mif"],
        M=cfg["M_global_2"],
        rw_sd=rw_sd_small,
        a=0.5,
        key=key
    )

    global_mif_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    key = jax.random.key(20260410)
    mod_global.pfilter(
        key=key,
        J=cfg["J_eval"],
        reps=cfg["eval_reps"]
    )
    global_eval_time = time.perf_counter() - t1

    pf = mod_global.results_history.last()

    rows = []
    for i in range(n_starts):
        lls = pf.logLiks.values[i, :]
        rows.append({
            **mod_global.theta[i],
            "loglik": pp.logmeanexp(lls),
            "loglik_se": pp.logmeanexp_se(lls)
        })

    global_results = pd.DataFrame(rows)
    global_results = global_results[np.isfinite(global_results["loglik"])]

    with open(cache_file, "wb") as f:
        pickle.dump(
            (global_results, global_mif_time, global_eval_time),
            f
        )

print(f"Global mif time: {global_mif_time:.2f} sec")
print(f"Global eval time: {global_eval_time:.2f} sec")
print(global_results.sort_values("loglik", ascending=False).head(10))

fig, axes = plt.subplots(2, 3, figsize=(9, 6))
pairs = [
    ("Beta", "loglik"),
    ("mu_EI", "loglik"),
    ("rho", "loglik"),
    ("eta", "iota"),
    ("Beta", "mu_EI"),
    ("eta", "loglik"),
]

for ax, (x, y) in zip(axes.flat, pairs):
    ax.scatter(global_results[x], global_results[y], s=10, alpha=0.6)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{cache_dir}/global_search_pairs.png", dpi=200)
plt.close()

best_idx = global_results["loglik"].idxmax()
best_fit = global_results.loc[best_idx]
print(best_fit)

# Question 7.2 (d)

# profile likelihood for reporting rate

from scipy.stats import chi2

rw_sd_rho_profile = pp.RWSigma(
    sigmas={
        "Beta": 0.02,
        "mu_EI": 0.02,
        "mu_IR": 0.0,
        "N": 0.0,
        "eta": 0.02,
        "iota": 0.02,
        "rho": 0.0,   # fix rho for the profile
        "k": 0.0
    },
    init_names=["eta", "iota"]
)

cache_file = f"{cache_dir}/profile-rho-seir.pkl"

if os.path.exists(cache_file):
    with open(cache_file, "rb") as f:
        profile_rho_df = pickle.load(f)
else:
    np.random.seed(12345)

    rho_grid = np.linspace(0.60, 0.82, 8)
    n_prof = 4
    profile_rows = []

    for i, rho_val in enumerate(rho_grid):
        theta_list = []
        for j in range(n_prof):
            th = theta.copy()
            th["rho"] = float(rho_val)
            th["Beta"] = np.random.uniform(5, 40)
            th["mu_EI"] = np.random.uniform(0.1, 1.5)
            th["eta"] = np.random.uniform(0.01, 0.2)
            th["iota"] = np.random.uniform(1e-5, 0.01)
            theta_list.append(th)

        mod = build_seir_model(theta_list)

        key = jax.random.key(1000 + i)
        mod.mif(J=250, M=8, rw_sd=rw_sd_rho_profile, a=0.5, key=key)

        key = jax.random.key(2000 + i)
        mod.pfilter(key=key, J=600, reps=2)
        pf = mod.results_history.last()

        for j in range(n_prof):
            lls = pf.logLiks.values[j, :]
            fp = mod.theta[j]
            profile_rows.append({
                "rho": rho_val,
                "Beta": fp["Beta"],
                "mu_EI": fp["mu_EI"],
                "eta": fp["eta"],
                "iota": fp["iota"],
                "loglik": pp.logmeanexp(lls),
                "loglik_se": pp.logmeanexp_se(lls)
            })

    profile_rho_df = pd.DataFrame(profile_rows)
    profile_rho_df = profile_rho_df[np.isfinite(profile_rho_df["loglik"])]

    with open(cache_file, "wb") as f:
        pickle.dump(profile_rho_df, f)

max_ll_rho = profile_rho_df["loglik"].max()
ci_cutoff_rho = max_ll_rho - 0.5 * chi2.ppf(0.95, df=1)

top_rho = (
    profile_rho_df
    .groupby(profile_rho_df["rho"].round(5))
    .apply(lambda g: g.nlargest(2, "loglik"), include_groups=False)
    .reset_index(drop=True)
)

rho_in_ci = top_rho[top_rho["loglik"] >= ci_cutoff_rho]

print(f"95% CI for rho: ({rho_in_ci['rho'].min():.3f}, {rho_in_ci['rho'].max():.3f})")

fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(top_rho["rho"], top_rho["loglik"], s=18)
ax.axhline(ci_cutoff_rho, color="red", linestyle="--")
ax.set_xlabel(r"$\rho$")
ax.set_ylabel("Profile log-likelihood")
ax.set_title("SEIR profile over reporting rate")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{cache_dir}/profile_rho.png", dpi=200)
plt.close()



