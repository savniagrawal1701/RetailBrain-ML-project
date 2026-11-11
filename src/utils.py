"""Small helpers shared by train.py and app.py."""

NUM_LABELS = {
    "distance_km": "distance (km)", "weight_kg": "package weight (kg)", "price": "order value",
    "freight": "shipping cost", "n_items": "number of items", "same_state": "seller and buyer in same state",
    "estimated_days": "promised delivery days",
    "first_price": "first order value", "first_freight": "first shipping cost",
    "first_n_items": "items in first order", "first_installments": "payment installments",
    "first_delivery_days": "first delivery time", "first_score": "first review score",
    "first_late": "first order arrived late", "sentiment_prob": "review sentiment",
    "has_review": "wrote a review",
}

_PREFIXES = [
    ("seller_state_", "seller in {}"), ("customer_state_", "customer in {}"),
    ("first_category_", "first bought: {}"), ("category_", "category: {}"),
    ("month_", "ordered in month {}"),
]


def pretty_feature(raw: str) -> str:
    n = raw.replace("num__", "").replace("cat__", "")
    for prefix, template in _PREFIXES:
        if n.startswith(prefix):
            return template.format(n[len(prefix):])
    return NUM_LABELS.get(n, n.replace("_", " "))


def r(x, nd=3):
    return round(float(x), nd)
