# Shelf-Level-Grocery-Sales-Forecasting-Inventory-Restock-Alert-Generation
This project implements an end-to-end supply chain solution for Corporacion Favorita, a large grocery retailer, by integrating demand forecasting with Generative AI inventory alert system. 

The system predicts shelf-level sales using specific item-store combinations and multiple forecasting approaches like Machine Learning, Deep Learning and Statistical Models. It then feeds these forecasts into an inventory simulation to calculate reorder points, safety stock and stockout risks. Finally, it leverages Google's Gemini LLM to generate actionable restock alerts in natural language for store managers, prioritizing perishable goods and lead-time constraints. 


## Key Features

### 1. Multi-model Forecasting
	- LightGBM: feature-rich gradient boosting based ensemble with more than 40 engineered features
	- SARIMA: statistical time series modeling for seasonal patterns
	- Prophet: Facebook's robust forecasting package for trend and seasonality
	- LSTM: deep learning technique for complex sequential dependencies
   
### 2. Advanced Feature Engineering
	- Temporal Features: day of week, month, holidays, seasonality
	- Lag Features: 1, 7, 14, 30 day historical sales
	- Rolling Statistics: moving averages and standard deviations
	- External Factors: oil prices, transactions, promotions
	- Product & Store Metadata: product perishability, product families, store clusters, store location
   
### 3. Stratified Sampling
   Efficient training on sample of data (10 stores and 10 items) ensuring representation of different product families and store clusters.
   
### 4. Inventory Simulation
  	An intelligent inventory simulation engine that calculates:
  	- Dynamic Safety Stock: based on demand variability (σ × √Lead Time)
  	- Reorder Points: dynamic thresholds considering lead time (Lead Time × Demand + Safety Stock)
  	- Lead Time Optimization: delivery schedules based on specific product families
  	- Service Level Targeting: 95% in-stock probability
  	- Perishability constraints

### 5.  AI powered Alert Generation
  	Uses Gemini Pro API to transform raw data into urgent, professional business alerts for making quick, time-efficient decisions in supply chain management.
  	- Context-Aware Recommendations: considers perishability, urgency, risk
  	- Actionable Insights: specific reorder quantities and timelines
  	- Fallback templates in case of API failure: reliable alerts in absence of API access


## Tech Stack

- Language: Python 3.11
- Data Manipulation: Pandas, NumPy, Py7zr
- Visualization: Matplotlib, Seaborn, Plotly
- Machine Learning: LightGBM, Scikit-Learn
- Time Series: Statsmodels (SARIMA), Prophet
- Deep Learning: TensorFlow / Keras (LSTM)
- Generative AI: Google Gemini Pro API


## Methodology & Steps

### 1. Data Preprocessing & EDA

- Ingested data from the Corporación Favorita Kaggle Competition dataset (Train, Items, Transactions, Holidays, Oil Prices, Stores).
- Performed stratified sampling to select item-store combinations ensuring family and cluster distributions.
- Conducted time-series decomposition and analysis of promotional lift (23% lift observed).


### 2. Feature Engineering

	Created a comprehensive set of features:
	- Temporal: day of week, month, holidays (national/regional/local).
	- Economic: oil price volatility and lag features.
	- History: sales lags (1, 7, 14, 30 days) and rolling window statistics (mean, std dev).


### 3. Model Selection & Training

Models were evaluated based on MAE, RMSE, and MAPE.
| Model | MAE | RMSE | MAPE |
| :--- | :--- | :--- | :--- |
| LightGBM | 14.80 | 30.03 | 55.18% |
| Prophet | 14.60 | 28.73 | 55.41% |
| LSTM | 20.18 | 37.39 | 68.43% |
| SARIMA | 23.73 | 44.42 | 70.08% |

LightGBM was selected for the final forecasting and alert system due to its balance of accuracy (lowest MAPE) and training efficiency. 

**Why were different approaches utilized in this project?**
This project compares a feature-rich machine learning model (LightGBM) against standard time series techniques like SARIMA, LSTM, and Prophet. The goal was to answer an important question: **Does the accuracy gained from complex feature engineering actually justify the overhead compared to using pure time series models?**

LightGBM with engineered features performs significantly better than pure time series methods for shelf-level grocery forecasting due to 
1. Multiple external factors (promotions, holidays, oil prices)
2. High product/store variability
3. Short-term tactical decisions needed
 

### 4. Inventory Alert System 

	The system simulates daily inventory depletion based on actual forecasts.
	- Main Logic: Triggers an alert when Inventory <= Reorder Point.
	- Perishability Logic: Perishable items have tighter safety stock buffers to reduce waste.
	- Lead Time Awareness: Calculates stockout risk percentages relative to supplier lead times (1-5 days depending on specific category).

**Alert Priority Levels**

| Priority | Condition | Action | Timeline |
| :--- | :--- | :--- | :--- |
| **CRITICAL** | Stockout before restock arrives | Order immediately (same day) | Immediate |
| **HIGH** | Stockout within 1.5$\times$ lead time | Order within 1-2 days | 1-2 days |
| **MEDIUM** | Below reorder point | Order within 3-5 days | 3-5 days |


**Sample Alert Output**
```
--------------------------------------------------
PERISHABLE ALERT #2 - 2017-04-28 - PRODUCE
Lead Time: 1.0 days | Reorder Qty: 549 units
--------------------------------------------------
HIGH INVENTORY REORDER ALERT - Store 45 (Quito)

Item: PRODUCE (#1503844)
Perishable: Yes

INVENTORY STATUS:
• Current Stock: 457 units ***BELOW REORDER POINT***
• Reorder Point: 546 units
• Safety Stock: 225 units
• Time Until Stockout: 1.4 days

DEMAND & SUPPLY CHAIN:
• Daily Forecast: 324 units
• Lead Time: 1.0 days
• Demand During Lead Time: 324 units
• Stockout Risk: 70.9%
PERISHABLE: Order only what can be sold within shelf life

RECOMMENDED ACTION:
ORDER NOW: 549 units
Timeline: Order within 0 days

This quantity accounts for:
- Forecast demand during 1.0-day lead time
- Safety stock for demand variability
- Perishability constraints
--------------------------------------------------
```


## Visualizations

The project generates a dashboard of insights including:
1. Stockout Risk Distribution: histogram of risk percentages across inventory.
2. Inventory vs. Demand: scatter plots identifying critical stock levels.
3. Potential Lost Sales: quantifying the economic impact of the alerts.


## Installation & Usage

**1. Clone the repository**
```
git clone https://github.com/yourusername/Shelf-Level-Grocery-Sales-Forecasting.git
cd Shelf-Level-Grocery-Sales-Forecasting
```

**2. Install dependencies**
```
pip install pandas numpy matplotlib seaborn plotly lightgbm tensorflow prophet google-generativeai py7zr
```

**3. Data Download**
- Download the [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) dataset.
- Place .7z files in the input/ directory.

**4. Configure API**
- Get a Google Gemini API Key.
- Set it in your environment variables or a .env file as GEMINI_API_KEY.

**5.  Run the notebook**
- Open shelf-level-grocery-sales-forecasting.ipynb and run all cells.


## Future Improvements

- **Hyperparameter Tuning**: use Optuna hyperparameter tuning to optimize LightGBM and LSTM parameters for maximum accuracy.
- **External Data**: integrate weather data into the forecasting model to sharpen predictions for weather-sensitive products like beverages and produce.
- **Deployment**: deploy the alert system within a simple, interactive Streamlit dashboard for instant, real-time user engagement.
