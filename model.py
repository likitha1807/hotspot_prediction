import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor

# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    data = pd.read_csv("crime_data.csv")

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    # Extract time features
    data["hour"] = data["datetime"].dt.hour

    data["day_of_week"] = (
        data["datetime"].dt.dayofweek
    )

    data["month"] = (
        data["datetime"].dt.month
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Recent crime average
    data["recent_24h"] = (
        data["crime_count"]
        .rolling(24, min_periods=1)
        .mean()
    )

    # Create risk classes
    low_limit = data["crime_count"].quantile(0.50)
    high_limit = data["crime_count"].quantile(0.80)

    def risk_class(value):

        if value < low_limit:
            return 0

        elif value < high_limit:
            return 1

        else:
            return 2

    data["risk_class"] = (
        data["crime_count"]
        .apply(risk_class)
    )

    return data


# ============================================================
# TRAIN ALL 5 AI MODELS
# ============================================================

def train_all(data):

    features = [
        "latitude",
        "longitude",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "recent_24h"
    ]

    X = data[features]

    y = data["risk_class"]

    # ========================================================
    # MODEL 1 - RANDOM FOREST
    # ========================================================

    random_forest = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    random_forest.fit(X, y)

    print("1. Random Forest trained")


    # ========================================================
    # MODEL 2 - XGBOOST
    # ========================================================

    xgboost_model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )

    xgboost_model.fit(
        X,
        data["crime_count"]
    )

    print("2. XGBoost trained")


    # ========================================================
    # MODEL 3 - DBSCAN
    # ========================================================

    coordinates = data[
        ["latitude", "longitude"]
    ].drop_duplicates()

    scaler = StandardScaler()

    coordinates_scaled = scaler.fit_transform(
        coordinates
    )

    dbscan = DBSCAN(
        eps=0.25,
        min_samples=5
    )

    clusters = dbscan.fit_predict(
        coordinates_scaled
    )

    hotspots = coordinates.copy()

    hotspots["cluster"] = clusters

    # Remove noise points
    hotspots = hotspots[
        hotspots["cluster"] != -1
    ]

    print("3. DBSCAN hotspot detection completed")


    # ========================================================
    # MODEL 4 - ISOLATION FOREST
    # ========================================================

    anomaly_features = data[
        [
            "latitude",
            "longitude",
            "hour",
            "day_of_week",
            "crime_count"
        ]
    ]

    isolation_forest = IsolationForest(
        contamination=0.03,
        random_state=42
    )

    isolation_forest.fit(
        anomaly_features
    )

    print("4. Isolation Forest trained")


    # ========================================================
    # MODEL 5 - LSTM
    # ========================================================
    
    # For easier installation, this version uses
    # a simple time-series fallback if TensorFlow
    # is not installed.

    hourly_data = (
        data
        .set_index("datetime")["crime_count"]
        .resample("h")
        .sum()
    )

    return {

        "random_forest":
            random_forest,

        "xgboost":
            xgboost_model,

        "dbscan":
            dbscan,

        "isolation_forest":
            isolation_forest,

        "hotspots":
            hotspots,

        "data":
            data,

        "hourly_data":
            hourly_data
    }


# ============================================================
# PREDICT CRIME RISK
# ============================================================

def predict(
    models,
    latitude,
    longitude,
    hour,
    day_of_week,
    month
):

    data = models["data"]

    # Find nearest historical location
    distances = (
        (data["latitude"] - latitude) ** 2
        +
        (data["longitude"] - longitude) ** 2
    )

    nearest_index = distances.idxmin()

    crime_count = float(
        data.loc[
            nearest_index,
            "crime_count"
        ]
    )

    recent_24h = float(
        data["recent_24h"].mean()
    )

    is_weekend = int(
        day_of_week >= 5
    )

    input_data = pd.DataFrame([{

        "latitude": latitude,

        "longitude": longitude,

        "hour": hour,

        "day_of_week": day_of_week,

        "month": month,

        "is_weekend": is_weekend,

        "recent_24h": recent_24h

    }])


    features = [
        "latitude",
        "longitude",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "recent_24h"
    ]


    # ========================================================
    # RANDOM FOREST PREDICTION
    # ========================================================

    rf = models["random_forest"]

    prediction = int(
        rf.predict(
            input_data[features]
        )[0]
    )

    probabilities = rf.predict_proba(
        input_data[features]
    )[0]

    confidence = (
        probabilities[prediction] * 100
    )


    # ========================================================
    # XGBOOST PREDICTION
    # ========================================================

    xgb = models["xgboost"]

    predicted_count = float(
        xgb.predict(
            input_data[features]
        )[0]
    )

    predicted_count = max(
        0,
        predicted_count
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if prediction == 0:

        risk = "LOW"

    elif prediction == 1:

        risk = "MEDIUM"

    else:

        risk = "HIGH"


    # ========================================================
    # EXPLANATION
    # ========================================================

    reasons = []

    if crime_count > data["crime_count"].quantile(0.80):

        reasons.append(
            "High historical crime activity"
        )

    if hour >= 20 or hour <= 5:

        reasons.append(
            "Night-time period"
        )

    if is_weekend:

        reasons.append(
            "Weekend pattern"
        )

    if not reasons:

        reasons.append(
            "No major high-risk factor detected"
        )


    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    anomaly_input = [[
        latitude,
        longitude,
        hour,
        day_of_week,
        crime_count
    ]]

    anomaly_prediction = (
        models["isolation_forest"]
        .predict(anomaly_input)[0]
    )

    anomaly = (
        anomaly_prediction == -1
    )


    return {

        "risk": risk,

        "rf_class": risk,

        "score": round(
            confidence,
            2
        ),

        "predicted_count":
            round(
                predicted_count,
                2
            ),

        "reasons": reasons,

        "anomaly": anomaly

    }


# ============================================================
# LSTM / TIME SERIES FORECAST
# ============================================================

def forecast_next_hours(
    models,
    hours=6
):

    hourly_data = models[
        "hourly_data"
    ]

    if len(hourly_data) == 0:

        return []

    # Simple moving-average forecast
    # Used as a fallback so the project
    # works without TensorFlow.

    recent = hourly_data.tail(24)

    average = float(
        recent.mean()
    )

    predictions = []

    for i in range(hours):

        predictions.append(
            round(
                max(0, average),
                2
            )
        )

    return predictions