# Restaurant Food Waste Predictor

A Flask machine-learning dashboard with an HTML/CSS-styled interface for
estimating restaurant food waste. It compares Linear Regression, Decision Tree,
and Random Forest regressors using a held-out test split and reports MAE, RMSE,
and R².

> The built-in demo dataset is synthetic. It is for demonstrating the workflow
> only and is not evidence of real-world model accuracy.

## Run locally

From this directory, install the packages and launch the Flask app. On Windows, use the Python launcher if `python` is not recognized:

```powershell
py -3 -m pip install -r requirements.txt
py -3 app.py
```

If Python is installed normally on your system, the equivalent commands are:

```powershell
python -m pip install -r requirements.txt
python app.py
```

The Flask app starts with synthetic demo data and is available at
<http://127.0.0.1:8765>. The dashboard has four pages—Overview, Prediction,
Model evaluation, and Dataset—served by the same Flask application. The
dashboard summarizes records and waste, compares models on a held-out test
split, and shows tree-model feature importance. After model selection, the
chosen regressor is refit on all labeled records for making predictions. To
train with your own records, upload a CSV on the Dataset page. Uploaded data is
shared across pages while the server is running and resets when the server
stops. **Download sample CSV** provides the expected column names and example
rows. Uploaded CSVs must be UTF-8 and no larger than 5 MB. The server binds to
localhost only and is intended for local use, not public deployment.

## CSV schema

Use one row per restaurant service/event. Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `food_type` | category | Food category |
| `guests` | number | Number of guests |
| `event_type` | category | Service or event type |
| `quantity_prepared_kg` | number | Food prepared, in kilograms |
| `storage_condition` | category | Storage quality/condition |
| `purchase_history_kg` | number | Historical amount purchased, in kilograms |
| `season` | category | Season |
| `preparation_method` | category | Preparation method |
| `location` | category | Geographic area/category |
| `price_per_kg` | number | Price per kilogram in the dataset's currency |
| `waste_kg` | number | Observed food waste, in kilograms (prediction target) |

Numeric inputs and the target must be zero or greater. Numeric input fields must
be present and valid; missing categorical values are imputed. Rows without a
target are excluded from training. The CSV can include additional columns; they
are not used by the model.

## Model evaluation notes

All models use the same 80/20 train/test split (fixed random seed). Categorical
features are one-hot encoded; missing categorical values are labeled `Unknown`.
The displayed best model is selected by lowest holdout MAE. For reliable
operational use, evaluate on representative historical data and consider
time-based validation if predicting future periods.

## Tests

```powershell
python -m pytest
```
