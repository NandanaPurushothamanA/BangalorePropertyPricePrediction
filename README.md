# PriceWise – Bengaluru House Price Prediction

A Streamlit web app that estimates house prices in Bengaluru from **location, total square feet, number of bathrooms and BHK**. The model is a Linear Regression trained on 13,000+ real property listings after extensive data cleaning and outlier removal.

**Tech stack:** Python · pandas · NumPy · scikit-learn · Matplotlib · Streamlit

## Dataset

Bengaluru House Data (Kaggle) – **13,320 listings**, 9 columns (area type, availability, location, size, society, total sqft, bath, balcony, price in lakhs).

## Approach

| Step | What was done |
|---|---|
| Feature selection | Dropped `area_type`, `society`, `balcony`, `availability`; kept location, size, total sqft, bath, price |
| Missing values | Removed rows with missing location, size or bath (13,320 → 13,246) |
| Feature engineering | Extracted `bhk` from text like "2 BHK" / "4 Bedroom"; converted sqft ranges like "2100 - 2850" to their average; created `price_per_sqft` |
| Location grouping | 1,293 locations → **242**, by grouping locations with ≤ 10 listings as "other" |
| Outlier removal | Removed listings with < 300 sqft per bedroom, price-per-sqft beyond 1 standard deviation of each location's mean, BHK price anomalies (e.g. a 3 BHK cheaper per sqft than a 2 BHK in the same location), and bathrooms > BHK + 1 |
| Encoding | One-hot encoded locations → final dataset of **7,251 rows × 244 features** |
| Model selection | Compared Linear Regression, Lasso and Decision Tree with `GridSearchCV` + 5-fold `ShuffleSplit` cross-validation |

## Results

| Model | Mean CV R² |
|---|---|
| **Linear Regression** | **0.818** |
| Decision Tree | 0.714 |
| Lasso | 0.687 |

Linear Regression was selected: **R² = 0.845** on the held-out test set and **0.82** average across 5 cross-validation splits.

## Project structure

```
├── priceprediction.ipynb            # data cleaning, EDA, outlier removal, model training
├── app1.py                          # Streamlit web app
├── bangalore_home_price_model.pickle  # trained model
├── columns.json                     # feature columns used by the app
├── Bengaluru_House_Data.csv         # dataset
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib streamlit
streamlit run app1.py
```

Select a location, enter the area, BHK and bathrooms, then click **Predict Price** to get the estimated price in lakhs.

## Possible improvements

- Try tree-based ensembles (Random Forest, XGBoost) for non-linear price patterns
- Predict log(price) to reduce the effect of very expensive properties
- Deploy the app on Streamlit Community Cloud

