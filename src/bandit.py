"""
Coupon agent (module 9) - a simulated store used to compare discount strategies.

There is no real coupon data in the Olist-style dataset, so we simulate how each
customer segment *might* react to a discount.  Every segment has a different
price sensitivity, which means there is no single best discount: the agent has
to learn it by trying things (explore) and then using what works (exploit).

Strategies compared (all see the exact same customers and the same random
"would they buy?" draws, so the comparison is fair):
    no_discount, flat_10, flat_20, random, epsilon_greedy, thompson, oracle
"""
import numpy as np

ARMS = [0, 5, 10, 20]          # discount in %
MARGIN = 0.35                  # gross margin before discount

# segment -> (base conversion, price elasticity, average order value)
PROFILES = {
    # loyal customers: a small nudge helps
    "Loyal Repeaters": (0.30, 0.45, 105.0),
    # big spenders buy anyway: discounts just burn margin
    "Big Spenders":    (0.18, 0.20, 153.0),
    # brand-new customers also convert without a coupon
    "New & Fresh":     (0.12, 0.05, 98.0),
    # dormant bargain-hunters only come back for a big deal
    "Dormant Savers":  (0.06, 3.00, 55.0),
}
SEGMENTS = list(PROFILES)
STRATEGIES = ["no_discount", "flat_10", "flat_20", "random", "epsilon_greedy", "thompson", "oracle"]


def conversion(seg, d):
    p0, e, _ = PROFILES[seg]
    return min(0.95, p0 * (1 + e * d / 10))


def unit_profit(seg, d):
    """Profit if the customer buys at discount d."""
    return PROFILES[seg][2] * (MARGIN - d / 100)


def expected_profit_table():
    return {s: [conversion(s, d) * unit_profit(s, d) for d in ARMS] for s in SEGMENTS}


def best_arm(seg):
    return int(np.argmax(expected_profit_table()[seg]))


def simulate(T=10000, seed=7, shares=None, eps=0.1, points=120):
    rng = np.random.default_rng(seed)
    shares = np.asarray(shares if shares is not None else [0.15, 0.2, 0.35, 0.3], float)
    shares = shares / shares.sum()
    seg_seq = rng.choice(len(SEGMENTS), T, p=shares)
    u_buy = rng.random(T)                       # shared randomness -> fair comparison
    u_arm = rng.random(T)
    conv = np.array([[conversion(s, d) for d in ARMS] for s in SEGMENTS])
    prof = np.array([[unit_profit(s, d) for d in ARMS] for s in SEGMENTS])
    exp_prof = conv * prof
    oracle_arm = exp_prof.argmax(1)
    flat10 = ARMS.index(10)

    # Beta posteriors (successes+1, failures+1) per segment/arm
    ts_a, ts_b = np.ones((4, 4)), np.ones((4, 4))
    eg_n, eg_sum = np.zeros((4, 4)), np.zeros((4, 4))
    ts_pulls = np.zeros((4, 4), int)
    ts_rng = np.random.default_rng(seed + 1)

    curves = {k: np.zeros(T) for k in STRATEGIES}
    for t in range(T):
        s = seg_seq[t]
        # choose arms
        a_ts = int(np.argmax(ts_rng.beta(ts_a[s], ts_b[s]) * prof[s]))
        if u_arm[t] < eps or eg_n[s].min() == 0:
            a_eg = int(u_arm[t] * 1e6) % 4 if u_arm[t] < eps else int(np.argmin(eg_n[s]))
        else:
            a_eg = int(np.argmax(eg_sum[s] / eg_n[s]))
        picks = {"no_discount": 0, "flat_10": flat10, "flat_20": 3, "random": int(u_arm[t] * 4) % 4,
                 "epsilon_greedy": a_eg, "thompson": a_ts, "oracle": int(oracle_arm[s])}
        for name, a in picks.items():
            bought = u_buy[t] < conv[s, a]
            r = prof[s, a] if bought else 0.0
            curves[name][t] = r
            if name == "thompson":
                ts_a[s, a] += bought; ts_b[s, a] += (not bought); ts_pulls[s, a] += 1
            elif name == "epsilon_greedy":
                eg_n[s, a] += 1; eg_sum[s, a] += r

    cum = {k: np.cumsum(v) for k, v in curves.items()}
    idx = np.linspace(0, T - 1, min(points, T)).astype(int)
    learned = {}
    for i, s in enumerate(SEGMENTS):
        post = ts_a[i] / (ts_a[i] + ts_b[i]) * prof[i]
        learned[s] = dict(learned_discount=ARMS[int(post.argmax())], true_best=ARMS[int(oracle_arm[i])],
                          pulls=ts_pulls[i].tolist(), share=float(shares[i]))
    totals = {k: float(v[-1]) for k, v in cum.items()}
    return dict(
        T=T, arms=ARMS, x=(idx + 1).tolist(),
        curves={k: np.round(v[idx], 2).tolist() for k, v in cum.items()},
        totals=totals,
        uplift_vs_flat_pct=round(100 * (totals["thompson"] / totals["flat_10"] - 1), 1),
        uplift_vs_none_pct=round(100 * (totals["thompson"] / totals["no_discount"] - 1), 1),
        uplift_vs_flat20_pct=round(100 * (totals["thompson"] / totals["flat_20"] - 1), 1),
        regret_pct=round(100 * (1 - totals["thompson"] / totals["oracle"]), 1),
        learned=learned,
    )
