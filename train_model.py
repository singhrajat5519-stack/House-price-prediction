import os
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "house_prices.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "area_sqft", "bedrooms", "bathrooms", "floors", "house_age", "city_distance_km"
]
CATEGORICAL_FEATURES = ["location_type", "parking"]
TARGET = "price"


def create_sample_dataset(path, n=1200):
    rng = np.random.default_rng(SEED)
    area = rng.integers(500, 4500, n)
    bedrooms = np.clip(np.round(area / 650 + rng.normal(0, 0.8, n)), 1, 7).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.65 + rng.normal(0, 0.5, n), 1), 1, 5)
    floors = rng.integers(1, 4, n)
    house_age = rng.integers(0, 45, n)
    distance = np.round(rng.uniform(1, 35, n), 1)
    location = rng.choice(["Urban", "Suburban", "Rural"], n, p=[0.45, 0.4, 0.15])
    parking = rng.choice(["Yes", "No"], n, p=[0.7, 0.3])

    location_bonus = np.select(
        [location == "Urban", location == "Suburban", location == "Rural"],
        [1800000, 900000, 0], default=0
    )
    parking_bonus = np.where(parking == "Yes", 350000, 0)
    price = (
        450000
        + area * 14500
        + bedrooms * 320000
        + bathrooms * 220000
        + floors * 180000
        - house_age * 26000
        - distance * 42000
        + location_bonus
        + parking_bonus
        + rng.normal(0, 450000, n)
    )
    price = np.maximum(price, 700000).round(0)

    df = pd.DataFrame({
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "house_age": house_age,
        "city_distance_km": distance,
        "location_type": location,
        "parking": parking,
        "price": price,
    })
    df.to_csv(path, index=False)
    return df


def load_data():
    if not os.path.exists(DATA_PATH):
        return create_sample_dataset(DATA_PATH)
    return pd.read_csv(DATA_PATH)


def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.20),
        Dense(64, activation="relu"),
        Dropout(0.15),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main():
    df = load_data()
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])

    X_train_processed = preprocessor.fit_transform(X_train).astype("float32")
    X_test_processed = preprocessor.transform(X_test).astype("float32")

    model = build_model(X_train_processed.shape[1])
    early_stop = EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    history = model.fit(
        X_train_processed,
        y_train.values,
        validation_split=0.2,
        epochs=250,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    predictions = model.predict(X_test_processed, verbose=0).flatten()
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    model.save(os.path.join(MODEL_DIR, "house_price_ann.keras"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    joblib.dump({
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "metrics": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
        "epochs_trained": len(history.history["loss"]),
    }, os.path.join(MODEL_DIR, "metadata.joblib"))

    print("\nTraining completed.")
    print(f"MAE:  ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")
    print(f"R²:   {r2:.4f}")
    print("Saved model files in models/")


if __name__ == "__main__":
    main()
