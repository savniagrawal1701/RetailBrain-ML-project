# RetailBrain

**An online store that sees its own problems coming.**

RetailBrain is an end-to-end machine learning project built around a fictional online store. It combines ten models that predict delivery times, late orders, review sentiment, repeat buyers, customer types, product recommendations, coupon decisions and weekly demand, and serves them through a Flask website with light and dark themes.

**Live demo:** _add your link here after deploying_

---

## 1. The problem

Every online store runs into the same issues:

- Parcels arrive later than promised, and customers get annoyed.
- Customers write reviews, but nobody has time to read thousands of them.
- Most first-time buyers never come back, and it is unclear who is worth contacting.
- All customers are treated alike, even though they behave very differently.
- Coupons are handed out blindly, which burns margin on people who would have bought anyway.
- Stock is ordered by guesswork instead of a forecast.

RetailBrain tackles each of these with a model that is trained on data and tested on data it has never seen.

## 2. What the project does

The project simulates a store with about 18,800 orders, 15,000 customers and 600 products. Ten models work on that shared data. Some feed into others, so it behaves like one system rather than ten separate experiments.

### Delivery
1. **Delivery time estimate.** Predicts how many days an order will take from distance, weight, value, shipping cost and month. Linear regression, Ridge and Lasso are compared.
2. **Late delivery risk.** Gives the probability that an order arrives after the promised date. Logistic regression is compared with a decision tree, random forest and XGBoost.
3. **Rules behind late orders.** A shallow decision tree turns the data into plain rules, for example "ordered in December, tight promise, long distance means over half are late".

### Customers
4. **Repeat buyer prediction.** Estimates which first-time customers will order again. Logistic regression, random forest, XGBoost and a tuned XGBoost are compared, using lift and recall because only a small share of customers return.
5. **Review sentiment.** Scores written reviews from unhappy to happy using TF-IDF features and Naive Bayes, and shows which words drove the score.
6. **Customer segments.** K-Means on recency, frequency and spend groups customers into Loyal Repeaters, Big Spenders, New & Fresh and Dormant Savers. PCA is used to visualise them.

### Recommendations and decisions
7. **Similar products.** K-nearest neighbours finds look-alike products by price, weight, rating and category.
8. **Recommendation engine.** A hybrid of collaborative filtering and popularity suggests what to show each customer, and handles new customers with little history.
9. **Coupon agent.** A Thompson-sampling bandit (a reinforcement learning idea) learns which discount suits each customer type by trying options and keeping what works, and is compared with fixed discount rules.
10. **Demand forecasting.** XGBoost and a neural network forecast weekly sales per category, compared with a simple "same as last week" baseline.

### How the models connect
- The sentiment model scores each customer's first review, and that score is an input to the repeat-buyer model.
- Late deliveries and review scores both help predict who comes back.
- The customer segments from K-Means are the groups the coupon agent learns about.
- The customer lookup page combines segment, repeat odds, coupon advice and recommendations.

## 3. Results

All scores are measured on held-out data.

| Task | Result |
|---|---|
| Delivery time | Off by 1.85 days on average (guessing the average gives 4.01), R² 0.75 |
| Late delivery | ROC-AUC 0.73; flagging the riskiest 10% of orders catches 35% of late ones |
| Repeat buyers | ROC-AUC 0.69; 1.96 times better than random at picking returners |
| Sentiment | 92% accuracy |
| Recommender | Hit rate of 31.6% in the top 10, versus 23.3% for "most popular" |
| Coupon agent | +15% profit versus no coupons, +8% versus 10% off for everyone |
| Forecast | XGBoost off by 19%, neural network 23%, "same as last week" 27% |

## 4. Toolkit

| Area | Tools |
|---|---|
| Language | Python 3.12 |
| Modelling | scikit-learn (regression, trees, forests, Naive Bayes, K-Means, KNN, PCA, MLP), XGBoost |
| Data | pandas, NumPy, SciPy |
| Backend | Flask, Gunicorn, a JSON API |
| Frontend | HTML, CSS, JavaScript with no framework, hand-drawn SVG charts, light and dark themes |
| Deployment | Docker, Render configuration, Procfile |
| Testing | pytest |
| Version control | Git and GitHub |

## 5. Advantages

- **End to end.** It covers data generation, training, evaluation, an API, a website, tests and deployment files, not just a notebook.
- **Many algorithms, each with a reason.** Every algorithm has a job and is compared against alternatives, so the choices are explained rather than assumed.
- **Honest evaluation.** Scores use held-out data, imbalance-aware metrics (ROC-AUC, PR-AUC, lift) and baselines, instead of accuracy alone.
- **Explainable.** The site shows why an order is risky, which words drove a review score, and what drives repeat purchases.
- **Models work together.** Outputs of one model feed another, like a real data product.
- **Fast and light.** Predictions return instantly, and the app runs on a small free server.
- **Easy to reproduce.** One command retrains everything in about 30 seconds, and the tests check that every model beats its baseline.
- **Polished interface.** Light and dark themes, mobile-friendly layout, no external chart library.

## 6. Disadvantages and limitations

- **The data is simulated.** It is generated to look like the Olist Brazilian marketplace, with cause and effect built in. The scores are a test of the methods, not findings about a real store.
- **Review text is templated.** The 92% sentiment accuracy is easier than real reviews, which have sarcasm, slang and mixed languages.
- **The coupon agent uses a hand-built store.** There is no real coupon data, so customer responses to discounts are assumed. Only the learning process is real.
- **Repeat-buyer scores are modest.** An ROC-AUC of 0.69 is useful for ranking but far from certain for any single customer.
- **Content similarity added nothing to the recommender.** Tuning gave it zero weight, so the final blend is collaborative filtering plus popularity.
- **The neural network struggles.** With limited data it loses to XGBoost on forecasting, so the four-week forecast on the site uses XGBoost.
- **No live data pipeline.** There is no monitoring for data drift, no automatic retraining and no database.
- **Small scale.** It is designed for a demo-sized dataset, not millions of orders.

## 7. Uses

**As a portfolio project**
- Shows supervised, unsupervised and reinforcement-style learning in one system.
- Demonstrates the full path from data to a deployed web application.

**As a template for real retail work**
- **Operations:** warn customers or switch carriers when an order is likely to be late.
- **Customer care:** sort reviews by sentiment and find the words behind complaints.
- **Marketing:** target the customers most likely to return and give coupons only where they pay off.
- **Merchandising:** show recommendations for new and existing customers, and group similar products.
- **Planning:** forecast demand by category to guide stock and staffing.

**As a learning tool**
- Compares algorithms on the same task, with tables and plots.
- Shows how to deal with imbalanced data, tune a model and avoid misleading metrics.

## 8. Run it locally

You need Python 3.12.

```bash
git clone https://github.com/savniagrawal1701/RetailBrain-ML-project.git
cd RetailBrain-ML-project
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000.

To retrain the models (about 30 seconds):

```bash
python train.py
```

To run the tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 9. Deploy

**Render:** push the repo to GitHub, choose *New → Blueprint* on Render, and select this repo. `render.yaml` handles the rest.

**Docker:**