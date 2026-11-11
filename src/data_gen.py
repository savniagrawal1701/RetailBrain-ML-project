"""
Synthetic Olist-style e-commerce data generator.

Why synthetic?  The real Olist dataset lives behind a Kaggle login, so this
project ships a generator that reproduces the *shape* of that data (orders,
customers, products, reviews, delivery dates) with realistic cause-and-effect
relationships baked in (distance -> delivery time, lateness -> bad reviews,
bad reviews -> fewer repeat buyers, ...).  Everything downstream (training,
the web app) only needs the DataFrames returned by `generate()`, so you can
swap in the real Olist tables later without touching the models.
"""
import numpy as np
import pandas as pd

START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-08-31")

# state: (lat, lon, seller_share, customer_share)
STATES = {
    "SP": (-23.55, -46.63, 0.50, 0.38), "RJ": (-22.91, -43.17, 0.06, 0.13),
    "MG": (-19.92, -43.94, 0.07, 0.12), "RS": (-30.03, -51.23, 0.05, 0.06),
    "PR": (-25.43, -49.27, 0.09, 0.06), "SC": (-27.59, -48.55, 0.07, 0.05),
    "BA": (-12.97, -38.50, 0.03, 0.04), "PE": (-8.05, -34.88, 0.02, 0.03),
    "DF": (-15.79, -47.88, 0.04, 0.04), "CE": (-3.73, -38.52, 0.02, 0.03),
    "GO": (-16.68, -49.25, 0.03, 0.03), "AM": (-3.12, -60.02, 0.02, 0.02),
}
REMOTE = {"AM", "CE", "PE", "BA"}

# category: (price_mu, weight_mu_kg, popularity, peak_month, season_amp)
CATEGORIES = {
    "bed_bath": (4.2, 1.6, 1.0, 6, 0.15), "health_beauty": (4.3, 0.5, 1.1, 5, 0.10),
    "sports_leisure": (4.4, 1.2, 0.8, 1, 0.20), "furniture_decor": (4.6, 4.0, 0.8, 9, 0.15),
    "computers_accessories": (4.7, 0.6, 0.7, 11, 0.35), "housewares": (4.2, 2.0, 0.9, 3, 0.10),
    "watches_gifts": (4.9, 0.3, 0.6, 12, 0.60), "telephony": (3.9, 0.3, 0.5, 11, 0.30),
    "garden_tools": (4.3, 3.0, 0.5, 10, 0.40), "auto": (4.4, 1.5, 0.5, 7, 0.10),
    "toys": (4.0, 0.9, 0.6, 12, 0.80), "perfumery": (4.3, 0.3, 0.5, 5, 0.45),
}
CAT_LIST = list(CATEGORIES)
STATE_LIST = list(STATES)

POS = ["great product", "arrived before the deadline", "loved it and will buy again",
       "excellent quality", "very fast delivery", "exactly as described",
       "very happy with this purchase", "good value for money", "i recommend this seller",
       "perfect, thank you", "super satisfied", "works great"]
NEG = ["the product never arrived on time", "terrible delivery delay", "poor quality, not as described",
       "very disappointed", "i waited too long", "i will not buy again", "the box was damaged",
       "worst purchase ever", "the seller ignored my messages", "product stopped working",
       "completely different from the photo", "awful experience"]
NEU = ["it is okay", "average product", "does the job", "nothing special",
       "delivery was fine", "as expected, nothing more", "so so", "decent but not amazing"]


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def state_distance(s1, s2):
    a, b = STATES[s1], STATES[s2]
    return float(haversine(a[0], a[1], b[0], b[1]))


def promised_days(distance_km, buffer=3.0):
    """Deterministic part of the delivery promise shown at checkout."""
    return int(round(6 + distance_km / 175 + buffer))


def _norm(w):
    w = np.asarray(w, float)
    return w / w.sum()


def _make_products(rng, n=600):
    pop = _norm([CATEGORIES[c][2] for c in CAT_LIST])
    cats = rng.choice(CAT_LIST, n, p=pop)
    rows = []
    for i, c in enumerate(cats):
        pm, wm, _, _, _ = CATEGORIES[c]
        rows.append(dict(
            product_id=f"P{i:04d}", category=c,
            price=round(float(np.exp(rng.normal(pm, 0.45))), 2),
            weight_kg=round(float(np.exp(rng.normal(np.log(wm), 0.5))), 2),
            rating=round(float(np.clip(rng.normal(4.1, 0.45), 2.8, 5.0)), 2),
            popularity=float(rng.pareto(1.6) + 0.2),
        ))
    return pd.DataFrame(rows)


def _sample_dates(rng, n):
    days = pd.date_range(START, END, freq="D")
    t = np.linspace(0, 1, len(days))
    w = 1 + 2.0 * t
    w = w * np.where(days.month == 11, 1.5, 1.0) * np.where(days.month == 12, 1.25, 1.0)
    w = w * np.where(days.dayofweek >= 5, 0.85, 1.0)
    idx = rng.choice(len(days), n, p=_norm(w))
    return days[idx] + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")


def _sample_categories(rng, dates):
    cats = np.empty(len(dates), dtype=object)
    months = np.asarray(dates.month)
    for m in np.unique(months):
        mask = months == m
        w = [CATEGORIES[c][2] * (1 + CATEGORIES[c][4] * np.cos(2 * np.pi * (m - CATEGORIES[c][3]) / 12))
             for c in CAT_LIST]
        cats[mask] = rng.choice(CAT_LIST, mask.sum(), p=_norm(w))
    return cats


def _build_orders(rng, cust, cust_idx, dates, cats, products):
    n = len(dates)
    prod_by_cat = {c: products[products.category == c] for c in CAT_LIST}
    pids, pprice, pweight = [], [], []
    for c in cats:
        sub = prod_by_cat[c]
        j = rng.choice(len(sub), p=_norm(sub.popularity.values))
        r = sub.iloc[j]
        pids.append(r.product_id); pprice.append(r.price); pweight.append(r.weight_kg)
    n_items = rng.choice([1, 2, 3], n, p=[0.85, 0.11, 0.04])
    price = np.round(np.array(pprice) * n_items, 2)
    weight = np.round(np.array(pweight) * n_items, 2)

    sstate = rng.choice(STATE_LIST, n, p=_norm([STATES[s][2] for s in STATE_LIST]))
    slat = np.array([STATES[s][0] for s in sstate]) + rng.normal(0, 0.6, n)
    slon = np.array([STATES[s][1] for s in sstate]) + rng.normal(0, 0.6, n)
    c = cust.iloc[cust_idx]
    dist = haversine(slat, slon, c.c_lat.values, c.c_lon.values)

    freight = np.clip(6 + 0.9 * weight + dist / 90 * 1.3 + rng.normal(0, 2.0, n), 4, None)
    handling = rng.gamma(2.0, 0.9, n)
    month = np.asarray(dates.month)
    season = np.where(month == 11, 2.2, 0) + np.where(month == 12, 2.8, 0)
    remote = np.isin(c.customer_state.values, list(REMOTE)) * 2.5
    noise = rng.gamma(1.5, 1.4, n) - 2.1 + np.where(rng.random(n) < 0.035, rng.exponential(6, n), 0)
    deliv = np.maximum(1, np.round(handling + 2 + dist / 190 + season + remote
                                   + 0.25 * weight ** 0.7 + noise)).astype(int)
    est = np.array([promised_days(d, b) for d, b in zip(dist, rng.uniform(0.0, 4.0, n))])
    late = (deliv > est).astype(int)

    mean = 4.4 - 1.9 * late - 0.05 * np.maximum(0, deliv - 12) + rng.normal(0, 0.8, n)
    score = np.clip(np.round(mean), 1, 5).astype(int)
    score = np.where(rng.random(n) < 0.04, 1, score)

    text = []
    for s in score:
        if rng.random() < 0.40:
            text.append("")
            continue
        pool = POS if s >= 4 else NEG if s <= 2 else None
        if pool is not None and rng.random() < 0.07:      # rating and words disagree (happens a lot in real reviews)
            pool = NEG if pool is POS else POS
        if pool is None:
            parts = [str(rng.choice(NEU)), str(rng.choice(POS + NEG))]
        else:
            parts = list(rng.choice(pool, rng.integers(1, 3), replace=False))
            if rng.random() < 0.25:                         # mixed feelings
                other = NEG if pool is POS else POS
                parts.append(("but " if pool is POS else "although ") + str(rng.choice(other)))
        text.append(" and ".join(parts))

    return pd.DataFrame(dict(
        customer_idx=cust_idx, customer_unique_id=c.customer_unique_id.values,
        customer_state=c.customer_state.values, seller_state=sstate,
        purchase_ts=dates, month=month, dow=np.asarray(dates.dayofweek),
        product_id=pids, category=cats, n_items=n_items, price=price, weight_kg=weight,
        freight=np.round(freight, 2),
        installments=rng.choice([1, 2, 3, 4, 6, 8, 10], n, p=[.45, .1, .12, .1, .1, .06, .07]),
        distance_km=np.round(dist, 1), same_state=(sstate == c.customer_state.values).astype(int),
        estimated_days=est, delivery_days=deliv, is_late=late,
        review_score=score, review_text=text,
    ))


def generate(seed=42, n_customers=15000):
    rng = np.random.default_rng(seed)
    products = _make_products(rng)

    cstate = rng.choice(STATE_LIST, n_customers, p=_norm([STATES[s][3] for s in STATE_LIST]))
    cust = pd.DataFrame(dict(
        customer_unique_id=[f"C{i:05d}" for i in range(n_customers)], customer_state=cstate,
        c_lat=np.array([STATES[s][0] for s in cstate]) + rng.normal(0, 0.8, n_customers),
        c_lon=np.array([STATES[s][1] for s in cstate]) + rng.normal(0, 0.8, n_customers),
    ))

    # ---- first order for every customer
    d1 = _sample_dates(rng, n_customers)
    first = _build_orders(rng, cust, np.arange(n_customers), d1, _sample_categories(rng, d1), products)

    # ---- who comes back?  (hidden "loyalty" trait the model can't see)
    loyalty = rng.normal(0, 0.5, n_customers)
    z = (-3.3 + 1.0 * (first.review_score >= 4) - 1.1 * (first.review_score <= 2)
         - 0.8 * first.is_late + 0.22 * np.log1p(first.price) - 0.2 * (first.installments >= 6)
         + 0.45 * first.category.isin(["health_beauty", "perfumery", "toys"])
         + 0.9 * ((first.review_score >= 4) & (first.price > 120))      # happy + big basket
         - 0.8 * ((first.is_late == 1) & (first.installments >= 4))     # late + paying slowly
         + loyalty)
    repeat = rng.random(n_customers) < 1 / (1 + np.exp(-z.values))

    # ---- follow-up orders
    rep_idx = np.where(repeat)[0]
    extra_idx = np.repeat(rep_idx, 1 + np.minimum(rng.poisson(0.5, len(rep_idx)), 3))
    d2 = pd.Series(pd.DatetimeIndex(first.purchase_ts.values[extra_idx]
                                    + pd.to_timedelta(rng.integers(15, 240, len(extra_idx)), unit="D")))
    keep = (d2 <= END).values
    extra_idx, d2 = extra_idx[keep], pd.DatetimeIndex(d2[keep])
    same = rng.random(len(extra_idx)) < 0.6
    cats2 = np.where(same, first.category.values[extra_idx], _sample_categories(rng, d2))
    extra = _build_orders(rng, cust, extra_idx, d2, cats2, products)

    orders = pd.concat([first, extra], ignore_index=True).sort_values("purchase_ts").reset_index(drop=True)
    orders.insert(0, "order_id", [f"O{i:06d}" for i in range(len(orders))])

    # ---- browse + purchase events for the recommender (taste = 2 favourite categories)
    pop_cat = _norm([CATEGORIES[c][2] for c in CAT_LIST])
    by_cat = {c: products.index[products.category == c].values for c in CAT_LIST}
    pw = products.popularity.values
    ev = []
    for u in range(n_customers):
        fav = rng.choice(CAT_LIST, 2, replace=False, p=pop_cat)
        k = int(np.clip(2 + rng.poisson(3), 2, 12))
        seen = set()
        for _ in range(k * 3):
            c = rng.choice(fav) if rng.random() < 0.75 else rng.choice(CAT_LIST, p=pop_cat)
            ids = by_cat[c]
            seen.add(int(rng.choice(ids, p=_norm(pw[ids]))))
            if len(seen) >= k:
                break
        ev.extend((u, p) for p in seen)
    interactions = pd.DataFrame(ev, columns=["customer_idx", "product_idx"])

    # ---- customer table: first-order snapshot + RFM
    ref = END + pd.Timedelta(days=1)
    g = orders.groupby("customer_idx")
    fo = orders.sort_values("purchase_ts").drop_duplicates("customer_idx").set_index("customer_idx")
    customers = pd.DataFrame(dict(
        customer_unique_id=cust.customer_unique_id, customer_state=cust.customer_state,
        first_price=fo.price, first_freight=fo.freight, first_n_items=fo.n_items,
        first_installments=fo.installments, first_late=fo.is_late,
        first_delivery_days=fo.delivery_days, first_score=fo.review_score,
        first_category=fo.category, first_text=fo.review_text,
        recency=(ref - g.purchase_ts.max()).dt.days, frequency=g.size(),
        monetary=g.price.sum().round(2),
    ))
    customers["repeat"] = (customers.frequency >= 2).astype(int)
    return orders, customers.reset_index(drop=True), products, interactions
