"""
RetailBrain - web app.

    python app.py                       # dev server on http://localhost:5000
    gunicorn app:app                    # production

All models are trained by `python train.py` (it runs automatically on first start
if ./models is empty).
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template, request

from src import bandit
from src.config import CATS, NUM_CLF, NUM_REG, REP_CATS, REP_NUM
from src.data_gen import STATES, promised_days, state_distance
from src.reco import hybrid_scores
from src.utils import pretty_feature

ROOT = Path(__file__).parent
MODELS = ROOT / "models"

if not (MODELS / "artifacts.json").exists():          # fresh clone: train once
    import train
    train.main()

app = Flask(__name__)
app.json.sort_keys = False

ART = json.load(open(MODELS / "artifacts.json"))
DELIVERY = joblib.load(MODELS / "delivery_time.joblib")
LATE = joblib.load(MODELS / "late_logistic.joblib")
SENT = joblib.load(MODELS / "sentiment_nb.joblib")
REPEAT = joblib.load(MODELS / "repeat_xgb.joblib")["pipe"]
KNN = joblib.load(MODELS / "knn_products.joblib")
RECO = joblib.load(MODELS / "recommender.joblib")
CUSTOMERS = pd.read_csv(MODELS / "customers.csv.gz").fillna({"first_text": ""})
PRODUCTS = pd.read_csv(MODELS / "products.csv")

PROMISE_BUFFER = 2.0          # average padding retailers add to the delivery promise
MIN_SAME_STATE_KM = 80.0      # seller and buyer in the same state are still ~80 km apart
SEG_ACTION = {
    "Loyal Repeaters": "A small nudge works: keep coupons light.",
    "Big Spenders": "They buy anyway. Skip the discount and protect margin.",
    "New & Fresh": "No coupon needed yet. Focus on a smooth first delivery.",
    "Dormant Savers": "Win them back with a strong offer; they respond to price.",
}


def dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def bad(msg, code=400):
    return jsonify(error=msg), code


def num(data, key, lo, hi, default=None):
    v = data.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number")
    if not lo <= v <= hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}")
    return v


# ------------------------------------------------------------------ pages
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/api/meta")
def meta():
    m = ART["m1_delivery_time"]; l = ART["m2_3_late"]["table"][0]
    return jsonify(
        states=list(STATES), categories=ART["meta"]["categories"], segments=ART["meta"]["segments"],
        dataset=ART["dataset"],
        headline=dict(delivery_mae=[x for x in m["table"] if x["model"] == m["best"]][0]["mae"],
                      delivery_r2=[x for x in m["table"] if x["model"] == m["best"]][0]["r2"],
                      late_auc=l["roc_auc"], sentiment_acc=ART["m5_sentiment"]["table"][0]["accuracy"],
                      repeat_lift=ART["m4_repeat"]["table"][-1]["lift_top20"]),
    )


# ------------------------------------------------------------------ modules 1-2: order oracle
@app.post("/api/order")
def order():
    d = request.get_json(silent=True) or {}
    try:
        seller, buyer, cat = d.get("seller_state"), d.get("customer_state"), d.get("category")
        if seller not in STATES or buyer not in STATES:
            raise ValueError("Pick a valid seller and buyer state")
        if cat not in ART["meta"]["categories"]:
            raise ValueError("Pick a valid category")
        weight = num(d, "weight_kg", 0.05, 60, 1)
        price = num(d, "price", 1, 5000, 80)
        items = int(num(d, "n_items", 1, 10, 1))
        month = int(num(d, "month", 1, 12, 6))
    except ValueError as e:
        return bad(str(e))

    dist = state_distance(seller, buyer)
    same = int(seller == buyer)
    if same:
        dist = MIN_SAME_STATE_KM
    freight = float(d["freight"]) if d.get("freight") not in (None, "") else max(4.0, 6 + 0.9 * weight + dist / 90 * 1.3)
    promised = int(d["promised_days"]) if d.get("promised_days") not in (None, "") else promised_days(dist, PROMISE_BUFFER)
    row = pd.DataFrame([dict(distance_km=dist, weight_kg=weight, price=price, freight=freight, n_items=items,
                             same_state=same, estimated_days=promised, seller_state=seller, customer_state=buyer,
                             category=cat, month=month)])

    pred = float(DELIVERY["pipe"].predict(row[NUM_REG + CATS])[0])
    days = max(1, round(pred))
    lo, hi = max(1, round(pred + DELIVERY["q10"])), max(1, round(pred + DELIVERY["q90"]))

    pipe = LATE["pipe"]
    p_late = float(pipe.predict_proba(row[NUM_CLF + CATS])[0, 1])
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    contrib = clf.coef_[0] * dense(pre.transform(row[NUM_CLF + CATS]))[0]
    names = [pretty_feature(n) for n in pre.get_feature_names_out()]
    # distance and the promised window are strongly correlated, so explain them as one net effect
    di, pi = names.index("distance (km)"), names.index("promised delivery days")
    contrib[di] += contrib[pi]; contrib[pi] = 0.0
    names[di] = "distance vs. the promised window"
    order_ = np.argsort(contrib)
    up = [dict(feature=names[i], effect=round(float(contrib[i]), 2)) for i in order_[::-1][:3] if contrib[i] > 0.05]
    down = [dict(feature=names[i], effect=round(float(contrib[i]), 2)) for i in order_[:2] if contrib[i] < -0.05]
    verdict = "high" if p_late >= LATE["thr_high"] else "medium" if p_late >= LATE["thr_med"] else "low"
    return jsonify(distance_km=round(dist), promised_days=promised, predicted_days=days, range=[lo, hi],
                   late_probability=round(p_late, 3), base_rate=round(LATE["base_rate"], 3), verdict=verdict,
                   pushes_up=up, pushes_down=down, freight=round(freight, 2))


# ------------------------------------------------------------------ module 5: sentiment
@app.post("/api/review")
def review():
    text = str((request.get_json(silent=True) or {}).get("text", "")).strip()[:600]
    if not text:
        return bad("Write a review first")
    vec, clf = SENT.named_steps["tfidf"], SENT.named_steps["clf"]
    X = vec.transform([text])
    idx = X.nonzero()[1]
    if len(idx) == 0:
        return jsonify(positive=0.5, label="unsure", stars=3.0, words=[], known_words=0)
    p = float(SENT.predict_proba([text])[0, 1])
    vocab = vec.get_feature_names_out()
    diff = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    cands = [{"word": vocab[i], "weight": round(float(diff[i] * X[0, i]), 3)} for i in idx]
    cands = [w for w in cands if abs(w["weight"]) >= 0.05]
    if sum(abs(w["weight"]) for w in cands) < 0.4:        # nothing the model has real evidence about
        return jsonify(positive=0.5, label="unsure", stars=3.0, words=[], known_words=int(len(idx)))
    single = [w for w in cands if " " not in w["word"]]
    words = sorted(single or cands, key=lambda w: -abs(w["weight"]))[:8]
    return jsonify(positive=round(p, 3), label="positive" if p >= 0.5 else "negative",
                   stars=round(1 + 4 * p, 1), words=words, known_words=int(len(idx)))


# ------------------------------------------------------------------ modules 4, 6, 7, 8: customers
def _customer_payload(i):
    c = CUSTOMERS.iloc[i]
    x = CUSTOMERS.iloc[[i]][REP_NUM + REP_CATS]
    pre, booster = REPEAT.named_steps["pre"], REPEAT.named_steps["xgb"].get_booster()
    contrib = booster.predict(xgb.DMatrix(dense(pre.transform(x))), pred_contribs=True)[0][:-1]
    names = [pretty_feature(n) for n in pre.get_feature_names_out()]
    top = np.argsort(-np.abs(contrib))[:4]
    reasons = [dict(feature=names[j], effect=round(float(contrib[j]), 2)) for j in top if abs(contrib[j]) > 0.02]

    hist = np.array(RECO["hist"].get(i, [0]))
    s = hybrid_scores(hist, RECO["collab"], RECO["content"], RECO["pop"])
    recs = []
    for j in np.argsort(-s)[:6]:
        p = PRODUCTS.iloc[j]
        recs.append(dict(product_id=p.product_id, category=p.category, price=round(float(p.price), 2),
                         rating=float(p.rating)))
    seen = [dict(product_id=PRODUCTS.iloc[j].product_id, category=PRODUCTS.iloc[j].category) for j in hist[:5]]
    seg = c.segment
    return dict(
        id=c.customer_unique_id, state=c.customer_state, segment=seg, action=SEG_ACTION.get(seg, ""),
        discount=ART["m9_bandit"]["learned"][seg]["learned_discount"],
        repeat_probability=round(float(c.repeat_prob), 3), base_rate=ART["dataset"]["repeat_rate"],
        reasons=reasons, recency=int(c.recency), orders=int(c.frequency), spend=round(float(c.monetary), 2),
        first_score=int(c.first_score), first_late=bool(c.first_late), first_category=c.first_category,
        history_len=int(len(hist)), seen=seen, recommendations=recs,
        strategy="warm start: collaborative + content + popularity" if len(hist) >= 3
        else "cold start: leaning on popularity and product similarity")


@app.get("/api/customer/<cid>")
def customer(cid):
    cid = cid.strip().upper()
    if cid.startswith("C") and cid[1:].isdigit():
        i = int(cid[1:])
    elif cid.isdigit():
        i = int(cid)
    else:
        return bad("Customer ids look like C00042")
    if not 0 <= i < len(CUSTOMERS):
        return bad(f"Customer ids run from C00000 to C{len(CUSTOMERS) - 1:05d}", 404)
    return jsonify(_customer_payload(i))


@app.get("/api/customers/random")
def customer_random():
    return jsonify(_customer_payload(int(np.random.randint(len(CUSTOMERS)))))


@app.get("/api/products")
def products():
    cat = request.args.get("category", "")
    sub = PRODUCTS[PRODUCTS.category == cat].sort_values("popularity", ascending=False).head(40)
    return jsonify([dict(product_id=r.product_id, price=round(float(r.price), 2), rating=float(r.rating),
                         weight_kg=float(r.weight_kg)) for r in sub.itertuples()])


@app.get("/api/similar/<pid>")
def similar(pid):
    hit = PRODUCTS.index[PRODUCTS.product_id == pid]
    if len(hit) == 0:
        return bad("Unknown product", 404)
    i = int(hit[0])
    dist, idx = KNN["nn"].kneighbors(KNN["X"][i:i + 1], n_neighbors=7)
    out = []
    for d, j in zip(dist[0], idx[0]):
        if j == i:
            continue
        p = PRODUCTS.iloc[j]
        out.append(dict(product_id=p.product_id, category=p.category, price=round(float(p.price), 2),
                        rating=float(p.rating), match=round(float(1 / (1 + d)), 2)))
    p = PRODUCTS.iloc[i]
    return jsonify(product=dict(product_id=p.product_id, category=p.category, price=round(float(p.price), 2),
                                rating=float(p.rating)), neighbors=out[:5])


# ------------------------------------------------------------------ module 9: pricing agent
@app.post("/api/pricing")
def pricing():
    d = request.get_json(silent=True) or {}
    try:
        T = int(num(d, "rounds", 1000, 30000, 20000))
        seed = int(num(d, "seed", 0, 10**6, 7))
    except ValueError as e:
        return bad(str(e))
    return jsonify(bandit.simulate(T=T, seed=seed, shares=ART["m9_bandit"]["shares"]))


# ------------------------------------------------------------------ module 10: forecast
@app.get("/api/forecast/<category>")
def forecast(category):
    F = ART["m10_forecast"]
    if category not in F["categories"]:
        return bad("Unknown category", 404)
    return jsonify(weeks=F["weeks"], test_start=F["test_start"], future_weeks=F["future_weeks"],
                   series=F["categories"][category], table=F["table"])


# ------------------------------------------------------------------ model lab
@app.get("/api/lab")
def lab():
    out = {k: v for k, v in ART.items() if k not in ("m10_forecast", "m9_bandit", "meta")}
    out["m10_forecast"] = dict(table=ART["m10_forecast"]["table"])
    out["m9_bandit"] = {k: ART["m9_bandit"][k] for k in
                        ("totals", "uplift_vs_flat_pct", "uplift_vs_none_pct", "uplift_vs_flat20_pct", "regret_pct", "T")}
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
