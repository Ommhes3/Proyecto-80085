from pathlib import Path
import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Configuración de InfluxDB desde .env
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG", "agro-iot")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "sensor_data")
MEASUREMENT = os.getenv("INFLUX_MEASUREMENT", "sensor_readings")


TARGET_VARIABLE = "temperature"

FEATURE_COLUMNS = [
    "humidity",
    "solar_radiation",
    "soil_ph",
    "crop",
    "zone",
    "condition"
]


def read_sensor_data_from_influx() -> pd.DataFrame:
    """
    Lee las lecturas históricas almacenadas en InfluxDB.
    """

    if not INFLUX_TOKEN:
        raise ValueError(
            "No se encontró INFLUX_TOKEN. Revisa el archivo .env en la raíz del proyecto."
        )

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> filter(fn: (r) => r._field == "value")
      |> filter(fn: (r) => exists r.parcel and exists r.crop and exists r.zone and exists r.condition and exists r.sensor)
      |> keep(columns: ["_time", "_value", "parcel", "crop", "zone", "condition", "sensor"])
    '''

    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )

    query_api = client.query_api()
    result = query_api.query_data_frame(query)

    client.close()

    if isinstance(result, list):
        if not result:
            raise ValueError("InfluxDB no retornó datos.")
        df = pd.concat(result, ignore_index=True)
    else:
        df = result

    if df.empty:
        raise ValueError("No se encontraron datos en InfluxDB para entrenar el modelo.")

    df = df.rename(columns={"_value": "value"})

    return df


def transform_influx_data_to_ml_dataset(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte lecturas individuales por sensor a una tabla por parcela y ventana de tiempo.

    De:
    time | parcel | sensor | value

    A:
    time | parcel | temperature | humidity | rainfall | solar_radiation | soil_ph | wind_speed
    """

    df = df_long.copy()
    df["_time"] = pd.to_datetime(df["_time"])

    # Agrupa lecturas cercanas en ventanas de 10 segundos
    df["time_group"] = df["_time"].dt.floor("10s")

    index_columns = [
        "time_group",
        "parcel",
        "crop",
        "zone",
        "condition"
    ]

    df_wide = df.pivot_table(
        index=index_columns,
        columns="sensor",
        values="value",
        aggfunc="last"
    ).reset_index()

    df_wide.columns.name = None

    return df_wide


def prepare_dataset(df_wide: pd.DataFrame):
    """
    Prepara X e y para predecir temperatura.

    y = temperature

    X = humidity, rainfall, solar_radiation, soil_ph, wind_speed,
        crop, zone, condition
    """

    df = df_wide.copy()

    if TARGET_VARIABLE not in df.columns:
        raise ValueError(f"No existe la columna objetivo: {TARGET_VARIABLE}")

    expected_numeric_columns = [
        "humidity",
        "rainfall",
        "solar_radiation",
        "soil_ph",
    ]

    for column in expected_numeric_columns:
        if column not in df.columns:
            df[column] = np.nan

    # Se eliminan filas sin temperatura, porque es la variable objetivo
    df = df.dropna(subset=[TARGET_VARIABLE])

    # Rellenar valores faltantes numéricos con mediana
    for column in expected_numeric_columns:
        median_value = df[column].median()

        if pd.isna(median_value):
            median_value = 0

        df[column] = df[column].fillna(median_value)

    # Eliminar filas sin variables categóricas importantes
    df = df.dropna(subset=["crop", "zone", "condition"])

    if len(df) < 10:
        raise ValueError(
            f"Hay muy pocos registros para entrenar el modelo de temperatura. Registros disponibles: {len(df)}"
        )

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_VARIABLE]

    # Convertir variables categóricas a numéricas
    X = pd.get_dummies(
        X,
        columns=["crop", "zone", "condition"],
        drop_first=True
    )

    return X, y, df


def train_random_forest(X: pd.DataFrame, y: pd.Series):
    """
    Entrena el modelo Random Forest.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_pred


def evaluate_model(y_test, y_pred):
    """
    Evalúa el modelo con MAE, RMSE y R2.
    """

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results = pd.DataFrame({
        "temperature_real": y_test.values,
        "temperature_predicted": y_pred
    })

    metrics = pd.DataFrame({
        "metric": ["MAE", "RMSE", "R2"],
        "value": [mae, rmse, r2]
    })

    print("\nRESULTADOS DEL MODELO RANDOM FOREST - TEMPERATURE")
    print("-" * 60)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    print("\nPrimeras predicciones:")
    print(results.head(10))

    return results, metrics


def show_feature_importance(model, X: pd.DataFrame):
    """
    Calcula la importancia de variables.
    """

    importance = pd.DataFrame({
        "variable": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nIMPORTANCIA DE VARIABLES")
    print("-" * 60)
    print(importance)

    return importance


def main():
    print("Leyendo datos históricos desde InfluxDB...")
    df_long = read_sensor_data_from_influx()

    print("\nDatos leídos desde InfluxDB:")
    print(df_long.head())

    print("\nTransformando datos para Machine Learning...")
    df_wide = transform_influx_data_to_ml_dataset(df_long)

    print("\nDataset transformado:")
    print(df_wide.head())

    print("\nPreparando variables para predecir temperatura...")
    X, y, df_model = prepare_dataset(df_wide)

    print("\nVariables de entrada:")
    print(X.columns.tolist())

    print("\nVariable objetivo:")
    print(TARGET_VARIABLE)

    print("\nEntrenando Random Forest...")
    model, X_train, X_test, y_train, y_test, y_pred = train_random_forest(X, y)

    results, metrics = evaluate_model(y_test, y_pred)
    importance = show_feature_importance(model, X)

    output_dir = PROJECT_ROOT / "Backend" / "MachineLearning" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_model.to_csv(output_dir / "temperature_ml_dataset_from_influx.csv", index=False)
    results.to_csv(output_dir / "temperature_predictions.csv", index=False)
    metrics.to_csv(output_dir / "temperature_metrics.csv", index=False)
    importance.to_csv(output_dir / "temperature_feature_importance.csv", index=False)

    versioned_model_path = output_dir / f"random_forest_temperature_model_{timestamp}.joblib"
    latest_model_path = output_dir / "random_forest_temperature_model_latest.joblib"

    joblib.dump(model, versioned_model_path)
    joblib.dump(model, latest_model_path)

    print("\nArchivos generados:")
    print(output_dir / "temperature_ml_dataset_from_influx.csv")
    print(output_dir / "temperature_predictions.csv")
    print(output_dir / "temperature_metrics.csv")
    print(output_dir / "temperature_feature_importance.csv")
    print(versioned_model_path)
    print(latest_model_path)


if __name__ == "__main__":
    main()