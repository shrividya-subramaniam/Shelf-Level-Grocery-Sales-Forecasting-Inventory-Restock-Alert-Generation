# Shelf-Level Grocery Sales Forecasting & Inventory Restock Alert Generation

An end-to-end demand forecasting and inventory replenishment system for Corporación Favorita, a large Ecuadorian grocery retailer, built on the [Kaggle Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting) dataset (125M+ rows across six tables).

The system forecasts daily sales for each item-store combination with statistical, machine learning and deep learning models, benchmarks them against simple baselines, and feeds the chosen forecast into an inventory simulation that places orders under lead times and perishability constraints. It measures what the forecast is worth by running the same inventory policy with different forecasts. Rule-based prioritisation turns reorders into alerts for store managers, and Google Gemini writes the most urgent ones in plain language.

## Key Results

All figures come from one full run on a stratified sample of 10 stores × 10 items (95 item-store series), with a test period of 112 days (26 Apr – 15 Aug 2017).

**Forecasting (one day ahead, 10,640 open-day test rows, same rows for every model)**

| Model | WAPE | RMSLE | MAE | RMSE |
|---|---|---|---|---|
| SARIMA | 48.0% | 0.604 | 2.13 | 3.91 |
| **LightGBM** | **48.2%** | **0.583** | 2.14 | **3.89** |
| LSTM | 53.1% | 0.650 | 2.36 | 4.18 |
| Prophet | 58.7% | 0.721 | 2.61 | 4.76 |
| *Baseline: 7-day moving average* | 51.8% | 0.621 | 2.30 | 4.16 |
| *Baseline: same day last week* | 62.4% | 0.787 | 2.77 | 5.01 |

LightGBM and SARIMA beat the stronger baseline (7-day moving average) by about 3.6 points of WAPE. LSTM and Prophet do not beat it.

**Inventory simulation: same policy, different forecasts**

| Forecast driving the policy | Fill rate | Stockout days | Lost units | Avg stock (units) |
|---|---|---|---|---|
| Training average (no forecast) | 96.4% | 3.3% | 1,690 | 43.6 |
| Same day last week | 97.2% | 2.8% | 1,346 | 50.7 |
| 7-day moving average | 97.3% | 2.8% | 1,272 | 44.8 |
| **LightGBM** | **97.4%** | **2.4%** | **1,206** | 46.0 |

LightGBM forecasts cut lost sales by **29%** against a no-forecast policy (for 5.5% more stock), and by 5.2% against the 7-day moving average. Perishables: 92.4% fill rate, 1.2% of stocked units expired.

**Alerts:** 1,939 reorders were placed over the test period. 1,198 routine reorders were logged without an alert, leaving **741 alerts** for store managers (240 CRITICAL, 501 HIGH).

## Key Features

### 1. Multi-model forecasting
- **LightGBM:** a single global gradient-boosting model with 42 engineered features
- **SARIMA:** one statistical model per item-store, with weekly seasonality
- **Prophet:** one model per item-store, with Ecuador's public holidays, the previous day's and previous week's sales, and the promotion flag as regressors
- **LSTM:** one recurrent neural network per item-store, reading the previous 30 days
- **Baselines:** same day last week, and the average of the previous 7 days

All models make one-day-ahead forecasts from actual sales up to the previous day, and are scored on the same rows.

### 2. Leak-free data preparation and feature engineering
- **Complete daily panel:** Favorita omits days with no sales, so every item-store gets one row per calendar day (12,722 zero-sales days added), with store-closed days flagged
- **Temporal:** day of week, month, quarter, week of year, weekend, month start and end
- **Sales history:** lags of 1, 7, 14 and 30 days; 7-, 14- and 30-day rolling means and standard deviations (all using only past days)
- **Holidays:** national, regional and local holidays matched to each store's state and city; transferred holidays and make-up workdays excluded; days to the nearest national holiday
- **External factors:** oil price and its changes and moving averages, and store transactions, all lagged by one day; promotions and lagged promotions
- **Product and store metadata:** product family, class, perishability, store cluster, type, city and state

### 3. Stratified sampling
10 stores and 10 items are drawn at random within strata: stores by cluster (9 clusters represented) and items by product family (DAIRY, DELI, GROCERY I, BEVERAGES, CLEANING). The sample is chosen from 2015 data only, before the 2016–2017 modelling period, so the test period plays no part in the selection.

### 4. Inventory simulation
A day-by-day simulation over the test period, depleted by **actual** sales:
- **Safety stock** = z × σ(daily demand) × √lead time, scaled by perishability and product family (z = 1.65 for a 95% service level)
- **Reorder point** = forecast daily demand × lead time + safety stock, recalculated every day from the forecast
- **Order quantity** = demand over the lead time + one review period (1 day for perishables, 7 days otherwise) + safety stock − (stock on hand + stock on order)
- **Lead times** by product family (1–5 days); orders arrive after the lead time
- **Perishables** are kept in dated batches, sold oldest first and thrown away after their shelf life (3–5 days)
- **Results:** fill rate, stockout days, lost units, waste rate and average stock

### 5. Alert prioritisation and AI-written alerts
- Each reorder is classified by how long the stock on hand plus stock on order will last compared with the supplier lead time
- Routine reorders are logged; only HIGH and CRITICAL reorders become alerts for store managers
- **Google Gemini 3.5 Flash**, with **Gemini 3.1 Flash-Lite** as fallback, writes the 10 most urgent alerts, with the exact reorder quantity, timeline and perishability guidance
- Requests are paced to stay under rate limits; temporary errors are retried with the delay the API suggests, and the code switches to the fallback model when a daily quota is used up
- If Gemini is unavailable, a deterministic template alert is used, so every alert still has a message

## Tech Stack

- **Language:** Python 3
- **Data manipulation:** Pandas, NumPy, py7zr
- **Visualisation:** Matplotlib, Seaborn
- **Machine learning:** LightGBM, Scikit-Learn
- **Time series:** statsmodels (SARIMA), Prophet
- **Deep learning:** TensorFlow / Keras (LSTM)
- **Generative AI:** Google Gemini API via the `google-genai` SDK (Gemini 3.5 Flash, Flash-Lite fallback)

## Methodology

### 1. Data preparation and EDA
- Loaded the Kaggle tables: train, items, stores, transactions, holidays and oil prices
- Drew the stratified sample, kept 2016–2017, clipped returns (negative sales) to zero and built the complete daily panel
- Explored sales by family, day of week and month; sales peak at weekends and in December
- Promoted item-days averaged 92.5% higher sales than non-promoted ones (a raw comparison, not a causal estimate)

### 2. Feature engineering
Built the 42 LightGBM features described above. Every feature uses only information available on the day before the forecast day.

### 3. Model training and evaluation
- **Split:** the first 80% of dates for training (up to 25 Apr 2017) and the last 20% for testing
- SARIMA, Prophet and LSTM are fitted on the training period only. LightGBM uses fixed hyperparameters (no tuning on the test set)
- **Metrics:** WAPE (share of units mis-forecast), RMSLE (log-scale error, the competition's metric), MAE, RMSE, and MAPE for reference
- Models are scored on open store-days where every model produced a forecast. 20 rows from one item-store that first sold after the split are excluded

**Why LightGBM?** It is effectively tied with SARIMA on WAPE (0.13 points apart). It was chosen for the lowest RMSLE and RMSE, and because it is one global model rather than 95 separate ones. Being global also lets it forecast item-stores it never saw in training, using their recent sales. A true cold start, with no sales history, is not tested.

### 4. Inventory simulation and alerts
The simulation runs every day of the test period, receiving orders, checking the reorder point and selling actual demand. The same simulation is then rerun with three simpler forecasts, so the difference in lost sales, stockouts, waste and stock held comes from the forecast alone.

## Alert Priority Levels

| Priority | Condition (stock on hand + on order) | Handling |
|---|---|---|
| CRITICAL | Runs out before the new order arrives (lasts ≤ lead time) | Alert: order today and ask the supplier to expedite |
| HIGH | Lasts ≤ 1.5 × lead time, or is at or below safety stock | Alert: order today, or within the days of slack left |
| MEDIUM | At or below the reorder point, but not at risk before delivery | Routine reorder, logged without an alert |

## Sample Alert Output

An alert from the final run. Gemini 3.5 Flash had reached its daily quota, so the fallback model, Gemini 3.1 Flash-Lite, wrote it:

```
--------------------------------------------------
PERISHABLE ALERT #2 - 2017-05-01 - DELI
Lead Time: 1 days | Reorder Qty: 3 units
--------------------------------------------------
**[CRITICAL INVENTORY ALERT] — CORPORACIÓN FAVORITA**

**To:** Store Manager — Store 5 (Santo Domingo)
**Product:** Item #671708 | Category: DELI (Perishable)
**Alert Status:** CRITICAL (Stockout)

### **Urgent Notice**
Item #671708 (DELI) is completely out of stock with zero units on hand or on order at Store 5,
breaching your reorder threshold and leaving zero hours of inventory coverage. Given the 1-day
supplier lead time, an immediate order must be submitted today to restore safety stock and prevent
prolonged sales loss while adhering to conservative ordering guidelines for perishables. Placing
this order now ensures replenishment arrives within the 24-hour lead-time window to meet expected demand.

### **Action Required**
* **CALCULATED REORDER QUANTITY: 3 units**

*Please approve and transmit this replenishment order immediately.*
--------------------------------------------------
```

## Visualisations

- **Model comparison:** metrics for all models and baselines, and the WAPE gain over each baseline
- **Policy comparison** (`policy_comparison.png`): lost units, stockout days and average stock for each forecast driving the same policy
- **Alert dashboard** (`alert_analysis_dynamic.png`):
  - lead-time demand as a share of stock at the time of each alert
  - stock against lead-time demand
  - alerts by demand-to-stock band
  - alerts by product family and by store
  - potential shortage

## Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/shrividya-subramaniam/Shelf-Level-Grocery-Sales-Forecasting-Inventory-Restock-Alert-Generation.git
cd Shelf-Level-Grocery-Sales-Forecasting
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn lightgbm statsmodels prophet tensorflow google-genai py7zr python-dotenv
```

**3. Get the data**
- The notebook is set up for Kaggle: add the competition dataset to the notebook, and it reads the `.7z` files from `/kaggle/input/competitions/favorita-grocery-sales-forecasting/`
- To run locally, download the dataset and change the paths in the data-loading cell to where you saved the `.7z` files

**4. Configure the Gemini API key** (optional; without a key, template alerts are used)
- Get a key from Google AI Studio
- **On Kaggle:** add it under Add-ons → Secrets as `GEMINI_API_KEY`
- **Locally:** set `GEMINI_API_KEY` as an environment variable or in a `.env` file

**5. Run the notebook**
Open `shelf_level_grocery_sales_forecasting_final.ipynb` and run all cells. The first step loads the full 125M-row training file, so allow plenty of memory (Kaggle's 30 GB is sufficient).

## Assumptions & Limitations

- **Assumed values:** lead times, shelf lives, safety-stock factors and opening stock are assumptions, as the dataset has no supply-chain data. The simulation results hold under these assumptions.
- **Sales as demand:** recorded sales are used as demand, so true demand on days that ran out of stock is under-counted.
- **Small sample, single split:** 10 stores × 10 items with one train/test split. Small differences between models, such as SARIMA vs LightGBM, are within noise.
- **Perishable stock is set too low:** perishables were out of stock on 11.1% of open days while only 1.2% expired, so their safety-stock factors should be raised.
- **Shorter LightGBM training period:** LightGBM's training starts about 30 days after the other models' (lag features need 30 days of history).

## Future Improvements

- **Hyperparameter tuning:** tune LightGBM and LSTM with Optuna on a time-based validation period (not the test set)
- **Stronger evaluation:** rolling-origin backtests over several periods, with confidence intervals
- **Better safety stock:** base it on forecast-error variability instead of demand variability, and tune the perishable factors
- **Cold-start test:** hold out item-stores entirely to test forecasting with no sales history
- **External data:** add weather data for weather-sensitive products such as beverages and produce
- **Deployment:** serve the alerts in an interactive Streamlit dashboard
