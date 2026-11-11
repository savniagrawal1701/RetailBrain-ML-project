"""Feature lists shared by training and serving (single source of truth)."""

CATS = ["seller_state", "customer_state", "category", "month"]
NUM_REG = ["distance_km", "weight_kg", "price", "freight", "n_items", "same_state"]
NUM_CLF = NUM_REG + ["estimated_days"]

REP_NUM = ["first_price", "first_freight", "first_n_items", "first_installments", "first_delivery_days",
           "first_score", "first_late", "sentiment_prob", "has_review"]
REP_CATS = ["customer_state", "first_category"]
