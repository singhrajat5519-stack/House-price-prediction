# House Price Prediction Using Deep Learning (ANN)

This project predicts house prices using a deep-learning Artificial Neural Network (ANN) and provides an interactive Streamlit UI.

## Project structure

```text
house_price_ann_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── house_prices.csv
├── models/
│   ├── house_price_ann.keras       # generated after training
│   ├── scaler.joblib               # generated after training
│   └── metadata.joblib             # generated after training
└── src/
    └── __init__.py
```

## Features

- Deep-learning regression using TensorFlow/Keras.
- Artificial Neural Network with dense layers, ReLU activation, dropout, and early stopping.
- Numerical and categorical feature preprocessing.
- Model evaluation using MAE, RMSE, and R².
- Interactive Streamlit prediction UI.
- Automatically creates a synthetic sample dataset if no dataset is supplied.

## Installation

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the ANN model

```bash
python train_model.py
```

The trained model and preprocessing files will be saved inside `models/`.

## Run the UI

```bash
streamlit run app.py
```

## Input features

- Area in square feet
- Number of bedrooms
- Number of bathrooms
- Number of floors
- House age
- Distance from city center in km
- Location type
- Parking availability

## Using your own dataset

Replace `data/house_prices.csv` with a CSV containing these columns:

```text
area_sqft,bedrooms,bathrooms,floors,house_age,city_distance_km,location_type,parking,price
```

`location_type` can contain values such as `Urban`, `Suburban`, or `Rural`; `parking` can contain `Yes` or `No`.
