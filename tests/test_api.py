"""Smoke tests: every endpoint answers, bad input is rejected politely, models beat their baselines."""
import pytest

from app import ART, app


@pytest.fixture(scope="module")
def client():
    return app.test_client()


ORDER = dict(seller_state="SP", customer_state="AM", category="toys", weight_kg=2, price=90, n_items=1, month=12)


def test_health_and_page(client):
    assert client.get("/health").get_json() == {"status": "ok"}
    assert b"RetailBrain" in client.get("/").data


def test_order_far_in_december_is_riskier_than_local_in_may(client):
    far = client.post("/api/order", json=ORDER).get_json()
    near = client.post("/api/order", json=dict(ORDER, customer_state="SP", month=5)).get_json()
    assert far["predicted_days"] > near["predicted_days"]
    assert far["late_probability"] > near["late_probability"]
    assert far["verdict"] in {"medium", "high"}


@pytest.mark.parametrize("patch", [dict(seller_state="XX"), dict(category="nope"), dict(weight_kg=-1), dict(month=13)])
def test_order_rejects_bad_input(client, patch):
    r = client.post("/api/order", json=dict(ORDER, **patch))
    assert r.status_code == 400 and "error" in r.get_json()


def test_review_sentiment(client):
    good = client.post("/api/review", json=dict(text="great product, very fast delivery")).get_json()
    bad = client.post("/api/review", json=dict(text="worst purchase ever, the box was damaged")).get_json()
    assert good["positive"] > 0.8 > 0.2 > bad["positive"]
    assert client.post("/api/review", json=dict(text="  ")).status_code == 400
    assert client.post("/api/review", json=dict(text="zzzz qqqq")).get_json()["label"] == "unsure"


def test_customer_lookup(client):
    j = client.get("/api/customer/C00042").get_json()
    assert j["segment"] in ART["meta"]["segments"] and 0 <= j["repeat_probability"] <= 1
    assert len(j["recommendations"]) == 6
    assert client.get("/api/customer/C99999").status_code == 404
    assert client.get("/api/customer/banana").status_code == 400
    assert client.get("/api/customers/random").status_code == 200


def test_similar_products_share_category(client):
    pid = client.get("/api/products?category=toys").get_json()[0]["product_id"]
    j = client.get(f"/api/similar/{pid}").get_json()
    assert len(j["neighbors"]) == 5
    assert sum(n["category"] == "toys" for n in j["neighbors"]) >= 4


def test_pricing_agent_beats_naive_rules(client):
    j = client.post("/api/pricing", json=dict(rounds=15000, seed=3)).get_json()
    assert j["totals"]["thompson"] > j["totals"]["no_discount"]
    assert j["totals"]["thompson"] > j["totals"]["flat_20"]
    assert client.post("/api/pricing", json=dict(rounds=5)).status_code == 400


def test_forecast(client):
    j = client.get("/api/forecast/toys").get_json()
    assert len(j["series"]["future"]["XGBoost"]) == 4
    assert client.get("/api/forecast/nope").status_code == 404


def test_models_beat_their_baselines():
    m1 = {r["model"]: r for r in ART["m1_delivery_time"]["table"]}
    assert m1["Ridge"]["mae"] < 0.6 * m1["Mean baseline"]["mae"]
    assert ART["m2_3_late"]["table"][0]["roc_auc"] > 0.65
    assert ART["m4_repeat"]["table"][-1]["lift_top20"] > 1.5
    assert ART["m5_sentiment"]["table"][0]["accuracy"] > 0.85
    reco = {r["model"]: r for r in ART["m8_recommender"]["table"]}
    assert reco["Hybrid"]["hit10"] > reco["Popularity"]["hit10"] > reco["Random"]["hit10"]
    fc = {r["model"]: r for r in ART["m10_forecast"]["table"]}
    assert fc["XGBoost"]["wape"] < fc["Naive (last week)"]["wape"]
