"""
RetailBrain - training pipeline.

    python train.py

Generates the data, trains all 10 modules, evaluates them, and writes everything
the web app needs into ./models and ./data/processed.  Takes ~1-2 minutes.
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score, silhouette_score)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier, XGBRegressor

from src import bandit
from src.config import CATS, NUM_CLF, NUM_REG, REP_CATS, REP_NUM
from src.reco import HYBRID_W, hybrid_scores, minmax
from src.data_gen import CAT_LIST, STATE_LIST, generate
from src.utils import pretty_feature, r

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
MODELS, DATA = ROOT / "models", ROOT / "data" / "processed"
MODELS.mkdir(exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
SEED = 0
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def make_pre(num, cats, scale):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
    ])


def topk(y, p, frac):
    """Of the `frac` highest-scored rows: share of all positives caught, and precision."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:k]
    caught = y[idx].sum()
    return float(caught / max(1, y.sum())), float(caught / k)


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# --------------------------------------------------------------------------------------
# Modules 1-3: delivery
# --------------------------------------------------------------------------------------
def module1_delivery_time(orders, tr, te):
    log("Module 1  Linear / Ridge / Lasso -> delivery time")
    Xtr, Xte = tr[NUM_REG + CATS], te[NUM_REG + CATS]
    ytr, yte = tr.delivery_days, te.delivery_days
    cands = {
        "Mean baseline": DummyRegressor(),
        "Linear Regression": LinearRegression(),
        "Ridge": RidgeCV(alphas=[0.1, 1, 10, 100]),
        "Lasso": LassoCV(cv=3, max_iter=5000, random_state=SEED),
    }
    rows, fitted = [], {}
    for name, est in cands.items():
        pipe = Pipeline([("pre", make_pre(NUM_REG, CATS, True)), ("m", est)]).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        row = dict(model=name, mae=r(mean_absolute_error(yte, pred), 2),
                   rmse=r(np.sqrt(mean_squared_error(yte, pred)), 2), r2=r(r2_score(yte, pred)))
        if name == "Ridge":
            row["note"] = f"alpha = {est.alpha_}"
        if name == "Lasso":
            row["note"] = f"kept {int((est.coef_ != 0).sum())} of {len(est.coef_)} features"
        rows.append(row); fitted[name] = pipe
    best = min((x for x in rows if x["model"] != "Mean baseline"), key=lambda x: x["mae"])["model"]
    pipe = fitted[best]
    resid = yte.values - pipe.predict(Xte)
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(pipe.named_steps["m"].coef_, index=[pretty_feature(n) for n in names])
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(8)
    joblib.dump(dict(pipe=pipe, q10=float(np.quantile(resid, 0.1)), q90=float(np.quantile(resid, 0.9))),
                MODELS / "delivery_time.joblib")
    return dict(table=rows, best=best, top_coefs=[dict(feature=k, coef=r(v, 2)) for k, v in top.items()],
                interval=[r(np.quantile(resid, 0.1), 2), r(np.quantile(resid, 0.9), 2)])


def cond_text(feat, thr, left):
    label = pretty_feature(feat)
    if feat.startswith("cat__"):
        return f"not {label}" if left else label
    return f"{label} <= {thr:.1f}" if left else f"{label} > {thr:.1f}"


def leaf_rules(pipe, Xtr, ytr, topn=4, min_n=50):
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    leaves = clf.apply(pre.transform(Xtr))
    stats = pd.DataFrame(dict(leaf=leaves, y=np.asarray(ytr))).groupby("leaf").y.agg(["size", "mean"])
    t, fn, rules = clf.tree_, pre.get_feature_names_out(), []

    def walk(node, conds):
        if t.children_left[node] == -1:
            if node in stats.index and stats.loc[node, "size"] >= min_n:
                rules.append(dict(conditions=conds, orders=int(stats.loc[node, "size"]),
                                  late_rate=r(stats.loc[node, "mean"])))
            return
        f, thr = fn[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [cond_text(f, thr, True)])
        walk(t.children_right[node], conds + [cond_text(f, thr, False)])

    walk(0, [])
    return sorted(rules, key=lambda x: -x["late_rate"])[:topn]


def module2_3_late(orders, tr, te):
    log("Module 2  Logistic Regression -> late delivery   |   Module 3  Decision Tree rules")
    cols = NUM_CLF + CATS
    Xtr, Xte, ytr, yte = tr[cols], te[cols], tr.is_late, te.is_late
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    models = {
        "Logistic Regression": Pipeline([("pre", make_pre(NUM_CLF, CATS, True)),
                                         ("clf", LogisticRegression(max_iter=3000))]),
        "Decision Tree": Pipeline([("pre", make_pre(NUM_CLF, CATS, False)),
                                   ("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=80,
                                                                  class_weight="balanced", random_state=SEED))]),
        "Random Forest": Pipeline([("pre", make_pre(NUM_CLF, CATS, False)),
                                   ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=10,
                                                                  class_weight="balanced_subsample", n_jobs=-1,
                                                                  random_state=SEED))]),
        "XGBoost": Pipeline([("pre", make_pre(NUM_CLF, CATS, False)),
                             ("clf", XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.06, subsample=0.9,
                                                   colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="logloss",
                                                   random_state=SEED))]),
    }
    rows, fitted = [], {}
    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        rec, prec = topk(yte, p, 0.10)
        rows.append(dict(model=name, roc_auc=r(roc_auc_score(yte, p)), pr_auc=r(average_precision_score(yte, p)),
                         recall_top10=r(rec), precision_top10=r(prec)))
        fitted[name] = pipe

    lr = fitted["Logistic Regression"]
    ptr = lr.predict_proba(Xtr)[:, 1]
    joblib.dump(dict(pipe=lr, thr_high=float(np.quantile(ptr, 0.90)), thr_med=float(np.quantile(ptr, 0.75)),
                     base_rate=float(ytr.mean())), MODELS / "late_logistic.joblib")

    tree = fitted["Decision Tree"]
    fn = tree.named_steps["pre"].get_feature_names_out()
    tree_text = export_text(tree.named_steps["clf"], feature_names=[pretty_feature(n) for n in fn], max_depth=4)
    imp = pd.Series(tree.named_steps["clf"].feature_importances_, index=[pretty_feature(n) for n in fn])
    imp = imp[imp > 0].sort_values(ascending=False).head(6)
    return dict(table=rows, base_rate=r(yte.mean()), tree_text=tree_text,
                tree_rules=leaf_rules(tree, Xtr, ytr),
                tree_importance=[dict(feature=k, importance=r(v)) for k, v in imp.items()])


# --------------------------------------------------------------------------------------
# Module 5: sentiment
# --------------------------------------------------------------------------------------
def module5_sentiment(orders, customers):
    log("Module 5  Naive Bayes -> review sentiment")
    lab = orders[(orders.review_text != "") & (orders.review_score != 3)]
    X, y = lab.review_text.values, (lab.review_score >= 4).astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)

    def tfidf():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    cands = {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.3))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=1000))]),
    }
    rows, fitted = [], {}
    for name, pipe in cands.items():
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        rows.append(dict(model=name, accuracy=r(accuracy_score(yte, p > 0.5)), f1=r(f1_score(yte, p > 0.5)),
                         roc_auc=r(roc_auc_score(yte, p))))
        fitted[name] = pipe
    nb = fitted["Naive Bayes"]
    joblib.dump(nb, MODELS / "sentiment_nb.joblib")
    vocab = np.array(nb.named_steps["tfidf"].get_feature_names_out())
    diff = nb.named_steps["clf"].feature_log_prob_[1] - nb.named_steps["clf"].feature_log_prob_[0]
    order = np.argsort(diff)

    # out-of-fold sentiment for each customer's first review (feeds module 4 without leakage)
    has_text = (customers.first_text != "") & (customers.first_score != 3)
    oof = np.full(len(customers), 0.5)
    sub = customers[has_text]
    oof[has_text.values] = cross_val_predict(clone(cands["Naive Bayes"]), sub.first_text.values,
                                             (sub.first_score >= 4).astype(int).values, cv=5,
                                             method="predict_proba")[:, 1]
    return dict(table=rows, n_reviews=int(len(lab)),
                top_positive=vocab[order[-8:]][::-1].tolist(), top_negative=vocab[order[:8]].tolist()), oof


# --------------------------------------------------------------------------------------
# Module 4: repeat buyers
# --------------------------------------------------------------------------------------
def module4_repeat(customers):
    log("Module 4  Logistic / Random Forest / XGBoost -> repeat buyers")
    X, y = customers[REP_NUM + REP_CATS], customers["repeat"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    spw = (ytr == 0).sum() / (ytr == 1).sum()

    def xgb_pipe(**kw):
        return Pipeline([("pre", make_pre(REP_NUM, REP_CATS, False)),
                         ("xgb", XGBClassifier(eval_metric="logloss", random_state=SEED, **kw))])

    models = {
        "Logistic Regression": Pipeline([("pre", make_pre(REP_NUM, REP_CATS, True)),
                                         ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("pre", make_pre(REP_NUM, REP_CATS, False)),
                                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=15,
                                                                  class_weight="balanced_subsample", n_jobs=-1,
                                                                  random_state=SEED))]),
        "XGBoost (class-weighted)": xgb_pipe(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, scale_pos_weight=spw),
    }
    search = RandomizedSearchCV(
        xgb_pipe(n_estimators=250), n_iter=10, cv=3, scoring="roc_auc", random_state=SEED, n_jobs=-1,
        param_distributions={"xgb__max_depth": [2, 3, 4, 5], "xgb__n_estimators": [150, 250, 400],
                             "xgb__learning_rate": [0.02, 0.04, 0.06, 0.1], "xgb__min_child_weight": [1, 5, 10, 20],
                             "xgb__subsample": [0.7, 0.85, 1.0], "xgb__colsample_bytree": [0.6, 0.8, 1.0],
                             "xgb__reg_lambda": [1, 5, 10]})
    rows = []
    for name, pipe in list(models.items()) + [("XGBoost (tuned)", search)]:
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        rec, prec = topk(yte, p, 0.20)
        rows.append(dict(model=name, roc_auc=r(roc_auc_score(yte, p)), pr_auc=r(average_precision_score(yte, p)),
                         recall_top20=r(rec), lift_top20=r(prec / yte.mean(), 2)))
    best = search.best_estimator_
    joblib.dump(dict(pipe=best, num=REP_NUM, cats=REP_CATS), MODELS / "repeat_xgb.joblib")

    # global importance via XGBoost's built-in SHAP values (no extra dependency)
    pre, booster = best.named_steps["pre"], best.named_steps["xgb"].get_booster()
    names = [pretty_feature(n) for n in pre.get_feature_names_out()]
    contrib = booster.predict(xgb.DMatrix(dense(pre.transform(Xte))), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=names).sort_values(ascending=False).head(8)
    return dict(table=rows, base_rate=r(yte.mean()), best_params={k.split("__")[1]: v for k, v in search.best_params_.items()},
                importance=[dict(feature=k, shap=r(v)) for k, v in imp.items()]), best


# --------------------------------------------------------------------------------------
# Module 6: segments
# --------------------------------------------------------------------------------------
def module6_segments(customers):
    log("Module 6  K-Means + PCA -> customer segments")
    raw = np.log1p(customers[["recency", "frequency", "monetary"]].values)
    Z = StandardScaler().fit_transform(raw)
    sil = {k: r(silhouette_score(Z, KMeans(k, n_init=5, random_state=SEED).fit_predict(Z),
                                 sample_size=4000, random_state=SEED)) for k in range(2, 8)}
    km = KMeans(4, n_init=10, random_state=SEED).fit(Z)
    lab = km.labels_
    stats = customers.assign(c=lab).groupby("c").agg(recency=("recency", "mean"), frequency=("frequency", "mean"),
                                                     monetary=("monetary", "mean"))
    names, left = {}, set(stats.index)
    for seg, key, fn in [("Loyal Repeaters", "frequency", "idxmax"), ("New & Fresh", "recency", "idxmin"),
                         ("Big Spenders", "monetary", "idxmax")]:
        c = getattr(stats.loc[list(left), key], fn)()
        names[c] = seg; left.discard(c)
    names[left.pop()] = "Dormant Savers"
    customers["segment"] = [names[c] for c in lab]

    prof = []
    for seg in bandit.SEGMENTS:
        s = customers[customers.segment == seg]
        prof.append(dict(segment=seg, customers=int(len(s)), share=r(len(s) / len(customers)),
                         recency_days=r(s.recency.mean(), 0), orders=r(s.frequency.mean(), 2),
                         spend=r(s.monetary.mean(), 0), repeat_rate=r(s["repeat"].mean())))
    pca = PCA(2, random_state=SEED).fit(Z)
    xy = pca.transform(Z)
    samp = np.random.default_rng(SEED).choice(len(Z), 1500, replace=False)
    scatter = [dict(x=r(xy[i, 0], 2), y=r(xy[i, 1], 2), s=customers.segment.iloc[i]) for i in samp]
    return dict(silhouette=sil, profiles=prof, pca_var=[r(v) for v in pca.explained_variance_ratio_],
                scatter=scatter)


# --------------------------------------------------------------------------------------
# Modules 7 + 8: similar products, recommender
# --------------------------------------------------------------------------------------
def product_matrix(products):
    num = np.column_stack([np.log(products.price), np.log(products.weight_kg), products.rating])
    num = StandardScaler().fit_transform(num)
    cat = pd.get_dummies(products.category).reindex(columns=CAT_LIST, fill_value=0).values * 1.5
    return np.hstack([num, cat])


def module7_knn(products):
    log("Module 7  KNN -> similar products")
    X = product_matrix(products)
    nn = NearestNeighbors(n_neighbors=7).fit(X)
    _, idx = nn.kneighbors(X)
    cats = products.category.values
    purity = float(np.mean([(cats[idx[i, 1:6]] == cats[i]).mean() for i in range(len(X))]))
    joblib.dump(dict(nn=nn, X=X), MODELS / "knn_products.joblib")
    return dict(category_purity_at5=r(purity), n_products=int(len(products)))


def module8_recommender(products, inter):
    log("Module 8  Hybrid recommender (collaborative + content + popularity blend)")
    from itertools import product as iproduct
    from scipy.sparse import csr_matrix
    n_items = len(products)
    content = cosine_similarity(product_matrix(products)).astype(np.float32)
    rng = np.random.default_rng(SEED)
    users = inter.groupby("customer_idx").product_idx.apply(list)
    chosen = rng.choice(users.index.values, 3000, replace=False)
    eval_users, tune_users = chosen[:2000], chosen[2000:]
    held = {u: users[u][int(rng.integers(len(users[u])))] for u in chosen}
    train = inter[[(held.get(u) != p) for u, p in zip(inter.customer_idx, inter.product_idx)]]

    def item_sim(frame):
        m = csr_matrix((np.ones(len(frame)), (frame.customer_idx, frame.product_idx)),
                       shape=(inter.customer_idx.max() + 1, n_items))
        s = cosine_similarity(m.T).astype(np.float32)
        np.fill_diagonal(s, 0)
        return s

    collab_eval = item_sim(train)
    pop = minmax(np.bincount(train.product_idx, minlength=n_items).astype(float))
    hist_eval = train.groupby("customer_idx").product_idx.apply(list)

    def hit(score, h, target):
        score = score.copy(); score[h] = -np.inf
        return float(target in np.argpartition(-score, 10)[:10])

    # --- tune blend weights on tune_users (never on the evaluation users)
    cache = []
    for u in tune_users:
        h = np.array(hist_eval[u])
        cache.append((h, held[u], minmax(collab_eval[:, h].sum(1)), minmax(content[:, h].mean(1))))
    grid = list(iproduct([0.3, 0.5, 0.7], [0.0, 0.1, 0.2], [0.1, 0.3, 0.5]))
    for grp, sel in [("warm", lambda h: len(h) >= 3), ("cold", lambda h: len(h) < 3)]:
        sub = [c for c in cache if sel(c[0])]
        best = max(grid, key=lambda w: np.mean([hit(w[0] * c1 + w[1] * c2 + w[2] * pop, h, t) for h, t, c1, c2 in sub]))
        HYBRID_W[grp] = tuple(float(x) for x in best)   # mutates src.reco.HYBRID_W in place

    # --- honest evaluation on held-out users
    hits = {k: [] for k in ["Random", "Popularity", "Content only", "Collaborative only", "Hybrid"]}
    cold = {k: [] for k in hits}
    for u in eval_users:
        h = np.array(hist_eval[u]); target = held[u]
        sc = {"Random": rng.random(n_items), "Popularity": pop, "Content only": minmax(content[:, h].mean(1)),
              "Collaborative only": minmax(collab_eval[:, h].sum(1)),
              "Hybrid": hybrid_scores(h, collab_eval, content, pop)}
        for k, sc_k in sc.items():
            v = hit(sc_k, h, target)
            hits[k].append(v)
            if len(h) == 1:
                cold[k].append(v)
    table = [dict(model=k, hit10=r(np.mean(v)), cold_hit10=r(np.mean(cold[k]))) for k, v in hits.items()]

    collab = item_sim(inter)
    pop_full = minmax(np.bincount(inter.product_idx, minlength=n_items).astype(float))
    hist = {int(u): v for u, v in inter.groupby("customer_idx").product_idx.apply(list).items()}
    joblib.dump(dict(collab=collab, content=content, pop=pop_full, hist=hist, weights=HYBRID_W),
                MODELS / "recommender.joblib")
    return dict(table=table, eval_users=len(eval_users), cold_users=len(cold["Hybrid"]),
                weights={k: list(v) for k, v in HYBRID_W.items()})


# --------------------------------------------------------------------------------------
# Module 10: demand forecasting
# --------------------------------------------------------------------------------------
def module10_forecast(orders):
    log("Module 10  XGBoost vs neural network (MLP) -> weekly demand")
    o = orders.assign(week=orders.purchase_ts.dt.to_period("W-SUN").dt.start_time)
    wk = o.pivot_table(index="week", columns="category", values="n_items", aggfunc="sum", fill_value=0)
    wk = wk[CAT_LIST].iloc[1:-1]                       # drop partial first/last week
    weeks, units = wk.index, wk.values.astype(float)
    L = np.log1p(units)
    nw, nc = L.shape
    TEST_W = 12

    def feats(Lm, t, c, week_start):
        woy = week_start.isocalendar()[1]
        oh = np.zeros(nc); oh[c] = 1
        return np.r_[[Lm[t - k, c] for k in (1, 2, 3, 4)], Lm[t - 4:t, c].mean(), Lm[max(0, t - 8):t, c].mean(),
                     np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52), t / nw, oh]

    rows, meta = [], []
    for t in range(8, nw):
        for c in range(nc):
            rows.append(feats(L, t, c, weeks[t])); meta.append((t, c))
    X, meta = np.array(rows), np.array(meta)
    y = np.array([L[t, c] for t, c in meta])
    tr, te = meta[:, 0] < nw - TEST_W, meta[:, 0] >= nw - TEST_W

    models = {
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                colsample_bytree=0.9, random_state=SEED),
        "Neural Net (MLP)": Pipeline([("sc", StandardScaler()),
                                      ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=0.003,
                                                           max_iter=1500, random_state=SEED))]),
    }
    preds = {"Naive (last week)": X[te][:, 0]}
    for name, m in models.items():
        m.fit(X[tr], y[tr]); preds[name] = m.predict(X[te])
    actual = np.expm1(y[te])
    table = []
    for name, p in preds.items():
        pu = np.expm1(p)
        table.append(dict(model=name, mae=r(np.abs(pu - actual).mean(), 2),
                          wape=r(np.abs(pu - actual).sum() / actual.sum())))

    # per-category series + 4-week recursive forecast
    HORIZON = 4
    future_weeks = [weeks[-1] + pd.Timedelta(weeks=k + 1) for k in range(HORIZON)]
    fut = {}
    for name, m in models.items():
        Le = np.vstack([L, np.zeros((HORIZON, nc))])
        for k in range(HORIZON):
            t = nw + k
            Le[t] = m.predict(np.array([feats(Le, t, c, future_weeks[k]) for c in range(nc)]))
        fut[name] = np.expm1(Le[nw:])
    test_weeks = sorted(set(meta[te][:, 0]))
    cat_out = {}
    for c, cat in enumerate(CAT_LIST):
        sel = meta[te][:, 1] == c
        cat_out[cat] = dict(
            history=[int(v) for v in units[:, c]],
            test={k: [round(float(v), 1) for v in np.expm1(p[sel])] for k, p in preds.items()},
            future={k: [round(float(v), 1) for v in fut[k][:, c]] for k in fut})
    return dict(table=table, weeks=[str(w.date()) for w in weeks], test_start=int(test_weeks[0]),
                future_weeks=[str(w.date()) for w in future_weeks], categories=cat_out)


# --------------------------------------------------------------------------------------
def main():
    log("Generating synthetic Olist-style data")
    orders, customers, products, inter = generate(seed=42)
    # random stratified split: a time split would leave the Nov/Dec peak out of the test window
    tr, te = train_test_split(orders, test_size=0.25, random_state=SEED, stratify=orders.is_late)

    A = {"dataset": dict(orders=int(len(orders)), customers=int(len(customers)), products=int(len(products)),
                         start=str(orders.purchase_ts.min().date()), end=str(orders.purchase_ts.max().date()),
                         late_rate=r(orders.is_late.mean()), repeat_rate=r(customers["repeat"].mean()),
                         avg_delivery_days=r(orders.delivery_days.mean(), 1),
                         train_orders=int(len(tr)), test_orders=int(len(te)))}
    A["m1_delivery_time"] = module1_delivery_time(orders, tr, te)
    A["m2_3_late"] = module2_3_late(orders, tr, te)
    A["m5_sentiment"], oof = module5_sentiment(orders, customers)

    customers["sentiment_prob"] = oof
    customers["has_review"] = (customers.first_text != "").astype(int)
    A["m4_repeat"], repeat_model = module4_repeat(customers)
    customers["repeat_prob"] = repeat_model.predict_proba(customers[REP_NUM + REP_CATS])[:, 1]

    A["m6_segments"] = module6_segments(customers)
    A["m7_knn"] = module7_knn(products)
    A["m8_recommender"] = module8_recommender(products, inter)

    log("Module 9  Thompson-sampling coupon agent")
    shares = [p["share"] for p in A["m6_segments"]["profiles"]]
    A["m9_bandit"] = bandit.simulate(T=20000, seed=7, shares=shares)
    A["m9_bandit"]["shares"] = shares
    A["m10_forecast"] = module10_forecast(orders)

    A["meta"] = dict(states=STATE_LIST, categories=CAT_LIST, segments=bandit.SEGMENTS)
    with open(MODELS / "artifacts.json", "w") as f:
        json.dump(A, f)
    customers.drop(columns=["first_text"]).assign(first_text=customers.first_text).to_csv(
        MODELS / "customers.csv.gz", index=False)
    products.to_csv(MODELS / "products.csv", index=False)
    orders.to_csv(DATA / "orders.csv", index=False)
    log("Done. Models + artifacts written to ./models")


if __name__ == "__main__":
    main()
