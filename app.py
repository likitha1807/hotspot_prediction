from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD CRIME DATA
# ============================================================

try:

    data = pd.read_csv("crime_data.csv")

    print("\nCrime data loaded successfully!")
    print("Columns:", list(data.columns))
    print("Number of records:", len(data))

except Exception as e:

    print("Error loading crime_data.csv:", e)

    data = pd.DataFrame({
        "latitude": [12.9716, 12.9352, 13.0068],
        "longitude": [77.5946, 77.6245, 77.5833],
        "datetime": [
            "2026-01-01 08:00:00",
            "2026-01-02 20:00:00",
            "2026-01-03 18:00:00"
        ],
        "crime_count": [25, 55, 35]
    })


# ============================================================
# CLEAN DATA
# ============================================================

data.columns = data.columns.str.strip().str.lower()

# Convert values to numeric

data["latitude"] = pd.to_numeric(
    data["latitude"],
    errors="coerce"
)

data["longitude"] = pd.to_numeric(
    data["longitude"],
    errors="coerce"
)

data["crime_count"] = pd.to_numeric(
    data["crime_count"],
    errors="coerce"
)


# ============================================================
# DATETIME
# ============================================================

if "datetime" in data.columns:

    data["datetime"] = pd.to_datetime(
        data["datetime"],
        errors="coerce"
    )

    data["hour"] = data["datetime"].dt.hour

else:

    data["hour"] = 12


# Remove invalid rows

data = data.dropna(
    subset=[
        "latitude",
        "longitude",
        "crime_count"
    ]
)


# ============================================================
# FEATURE CREATION
# ============================================================

X = data[
    [
        "latitude",
        "longitude",
        "hour"
    ]
]

y = data["crime_count"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

if len(data) >= 5:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

else:

    X_train = X
    X_test = X
    y_train = y
    y_test = y


# ============================================================
# FIVE AI / ML MODELS
# ============================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Linear Regression": LinearRegression(),

    "KNN": KNeighborsRegressor(
        n_neighbors=min(3, len(X_train))
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}


# ============================================================
# TRAIN ALL MODELS
# ============================================================

trained_models = {}

for name, model in models.items():

    try:

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

        print(name, "trained successfully.")

    except Exception as e:

        print(
            name,
            "training failed:",
            e
        )


print("\nAll available AI models:")
print(list(trained_models.keys()))


# ============================================================
# MAIN PREDICTION MODEL
# ============================================================

if "Random Forest" in trained_models:

    main_model = trained_models["Random Forest"]

else:

    main_model = list(
        trained_models.values()
    )[0]


# ============================================================
# CREATE CRIME HOTSPOT MAP
# ============================================================

def build_map():

    # Center map around average coordinates

    center_lat = data["latitude"].mean()
    center_lon = data["longitude"].mean()

    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=11,
        tiles="OpenStreetMap"
    )


    # --------------------------------------------------------
    # HEATMAP DATA
    # --------------------------------------------------------

    heat_data = []

    for _, row in data.iterrows():

        heat_data.append(
            [
                float(row["latitude"]),
                float(row["longitude"]),
                float(row["crime_count"])
            ]
        )


    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    if len(heat_data) > 0:

        HeatMap(
            heat_data,
            radius=30,
            blur=25,
            max_zoom=13
        ).add_to(m)


    # --------------------------------------------------------
    # CRIME LOCATION MARKERS
    # --------------------------------------------------------

    for _, row in data.iterrows():

        count = float(
            row["crime_count"]
        )


        # Risk colour

        if count >= 60:

            color = "red"
            risk = "High Risk"

        elif count >= 30:

            color = "orange"
            risk = "Medium Risk"

        else:

            color = "green"
            risk = "Low Risk"


        popup_text = f"""
        <b>Crime Hotspot</b><br>
        Latitude: {row['latitude']:.4f}<br>
        Longitude: {row['longitude']:.4f}<br>
        Crime Count: {count:.0f}<br>
        Risk Level: {risk}
        """


        folium.CircleMarker(

            location=[
                row["latitude"],
                row["longitude"]
            ],

            radius=7,

            popup=folium.Popup(
                popup_text,
                max_width=300
            ),

            color=color,

            fill=True,

            fill_color=color,

            fill_opacity=0.7

        ).add_to(m)


    # --------------------------------------------------------
    # MAP LEGEND
    # --------------------------------------------------------

    legend_html = """

    <div style="
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 190px;

        background-color: white;

        border: 2px solid grey;

        z-index: 9999;

        font-size: 14px;

        padding: 12px;

        box-shadow: 0 2px 6px rgba(0,0,0,0.3);

        border-radius: 8px;
    ">

        <b>🚨 Crime Risk Level</b>

        <br><br>

        <span style="color:green;">
            ●
        </span>

        Low Risk

        <br>

        <span style="color:orange;">
            ●
        </span>

        Medium Risk

        <br>

        <span style="color:red;">
            ●
        </span>

        High Risk

    </div>

    """

    m.get_root().html.add_child(
        folium.Element(
            legend_html
        )
    )


    # IMPORTANT:
    # Use get_root().render()
    # NOT m.repr_html()

    return m.get_root().render()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    prediction = None

    confidence = None

    predicted_count = None

    latitude = None

    longitude = None

    crime_count = None

    hour = None


    # --------------------------------------------------------
    # HANDLE FORM SUBMISSION
    # --------------------------------------------------------

    if request.method == "POST":

        try:

            # Get values from HTML form

            latitude = float(
                request.form.get(
                    "latitude",
                    0
                )
            )

            longitude = float(
                request.form.get(
                    "longitude",
                    0
                )
            )

            crime_count = float(
                request.form.get(
                    "crime_count",
                    0
                )
            )

            hour = int(
                request.form.get(
                    "hour",
                    12
                )
            )


            # ------------------------------------------------
            # CREATE INPUT FOR AI MODEL
            # ------------------------------------------------

            input_data = pd.DataFrame({

                "latitude": [
                    latitude
                ],

                "longitude": [
                    longitude
                ],

                "hour": [
                    hour
                ]

            })


            # ------------------------------------------------
            # PREDICT CRIME COUNT
            # ------------------------------------------------

            predicted_count = float(
                main_model.predict(
                    input_data
                )[0]
            )


            # Prevent negative prediction

            predicted_count = max(
                0,
                predicted_count
            )


            # ------------------------------------------------
            # RISK CLASSIFICATION
            # ------------------------------------------------

            if predicted_count >= 60:

                prediction = "High Risk"

                confidence = min(
                    95,
                    70 + predicted_count / 10
                )


            elif predicted_count >= 30:

                prediction = "Medium Risk"

                confidence = min(
                    90,
                    55 + predicted_count / 10
                )


            else:

                prediction = "Low Risk"

                confidence = min(
                    90,
                    60 + predicted_count / 10
                )


            confidence = round(
                confidence,
                1
            )


            print("\nPrediction completed!")

            print(
                "Location:",
                latitude,
                longitude
            )

            print(
                "Hour:",
                hour
            )

            print(
                "Predicted crime count:",
                predicted_count
            )

            print(
                "Risk:",
                prediction
            )


        except Exception as e:

            print(
                "Prediction error:",
                e
            )

            prediction = "Unable to Predict"

            confidence = 0

            predicted_count = 0


    # --------------------------------------------------------
    # BUILD MAP
    # --------------------------------------------------------

    map_html = build_map()


    # --------------------------------------------------------
    # SEND DATA TO HTML
    # --------------------------------------------------------

    return render_template(

        "index.html",

        map_html=map_html,

        total_crimes=len(data),

        predictions_count=1
        if prediction
        else 0,

        ai_models=len(
            trained_models
        ),

        prediction=prediction,

        confidence=confidence,

        predicted_count=round(
            predicted_count,
            2
        )
        if predicted_count is not None
        else None,

        latitude=latitude,

        longitude=longitude,

        crime_count=crime_count,

        hour=hour

    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print(
        "=========================================="
    )

    print(
        "   AI CRIME HOTSPOT PREDICTION SYSTEM"
    )

    print(
        "=========================================="
    )

    print(
        "Open this URL in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "=========================================="
    )

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )