from pathlib import Path
import sys
import os
import time
from datetime import datetime, timezone

import pandas as pd
import joblib

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS


# =========================
# CONFIGURACIÓN GENERAL
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG", "agro-iot")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "sensor_data")

SOURCE_MEASUREMENT = os.getenv("INFLUX_MEASUREMENT", "sensor_readings")
PREDICTION_MEASUREMENT = "prediction_results"

PUBLISH_INTERVAL_SECONDS = 300  # 5 minutos


TEMPERATURE_MODEL_PATH = (
    PROJECT_ROOT
    / "Backend"
    / "MachineLearning"
    / "outputs"
    / "random_forest_temperature_model_latest.joblib"
)

RAINFALL_MODEL_PATH = (
    PROJECT_ROOT
    / "Backend"
    / "MachineLearning"
    / "outputs"
    / "random_forest_rainfall_rice_model_latest.joblib"
)


TEMPERATURE_REQUIRED_SENSORS = [
    "humidity",
    "solar_radiation",
    "soil_ph",
]

RAINFALL_REQUIRED_SENSORS = [
    "temperature",
    "humidity",
    "solar_radiation",
    "soil_ph",
]


# =========================
# VALIDACIONES
# =========================

def validate_environment():
    if not INFLUX_TOKEN:
        raise ValueError("No se encontró INFLUX_TOKEN. Revisa el archivo .env.")

    if not TEMPERATURE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo de temperatura: {TEMPERATURE_MODEL_PATH}. "
            "Primero ejecuta random_forest_temperature.py"
        )

    if not RAINFALL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo de rainfall: {RAINFALL_MODEL_PATH}. "
            "Primero ejecuta random_forest_rainfall.py"
        )


# =========================
# CONSULTAS A INFLUXDB
# =========================

def query_influx_dataframe(query: str) -> pd.DataFrame:
    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )

    try:
        result = client.query_api().query_data_frame(query)
    finally:
        client.close()

    if isinstance(result, list):
        if not result:
            return pd.DataFrame()
        return pd.concat(result, ignore_index=True)

    return result if result is not None else pd.DataFrame()


def get_available_parcels() -> pd.DataFrame:
    """
    Obtiene las parcelas activas y su cultivo desde InfluxDB.
    Retorna columnas: parcel, crop.
    """

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "{SOURCE_MEASUREMENT}")
      |> filter(fn: (r) => r._field == "value")
      |> filter(fn: (r) => exists r.parcel and exists r.crop)
      |> keep(columns: ["_time", "_value", "parcel", "crop"])
    '''

    df = query_influx_dataframe(query)

    if df.empty:
        return pd.DataFrame(columns=["parcel", "crop"])

    if "parcel" not in df.columns or "crop" not in df.columns:
        print("No se encontraron las columnas parcel y crop en la consulta.")
        print("Columnas recibidas:", df.columns.tolist())
        return pd.DataFrame(columns=["parcel", "crop"])

    df["parcel"] = df["parcel"].astype(str).str.strip()
    df["crop"] = df["crop"].astype(str).str.strip().str.lower()

    parcels_df = (
        df[["parcel", "crop"]]
        .dropna()
        .drop_duplicates()
        .sort_values("parcel")
        .reset_index(drop=True)
    )

    return parcels_df


def read_latest_sensor_values(parcel: str, crop_filter: str | None = None) -> dict:
    """
    Lee la última lectura disponible por sensor para una parcela.
    Si crop_filter se envía, filtra también por cultivo.
    """

    crop_line = ""
    if crop_filter:
        crop_line = f'|> filter(fn: (r) => r.crop == "{crop_filter}")'

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "{SOURCE_MEASUREMENT}")
      |> filter(fn: (r) => r._field == "value")
      |> filter(fn: (r) => r.parcel == "{parcel}")
      {crop_line}
      |> filter(fn: (r) => exists r.sensor and exists r.crop and exists r.zone and exists r.condition)
      |> group(columns: ["sensor"])
      |> last()
      |> keep(columns: ["_time", "_value", "sensor", "parcel", "crop", "zone", "condition"])
    '''

    df = query_influx_dataframe(query)

    if df.empty:
        raise ValueError(f"No hay datos recientes para la parcela {parcel}.")

    latest_data = {}

    for _, row in df.iterrows():
        sensor = row["sensor"]
        value = row["_value"]
        latest_data[sensor] = value

    latest_data["crop"] = str(df.iloc[0]["crop"]).strip().lower()
    latest_data["zone"] = df.iloc[0]["zone"]
    latest_data["condition"] = df.iloc[0]["condition"]

    return latest_data


# =========================
# PREPARACIÓN PARA MODELOS
# =========================

def prepare_temperature_input(input_data: dict, model) -> pd.DataFrame:
    missing_sensors = [
        sensor for sensor in TEMPERATURE_REQUIRED_SENSORS
        if sensor not in input_data
    ]

    if missing_sensors:
        raise ValueError(
            f"Faltan sensores para predecir temperatura: {missing_sensors}"
        )

    df = pd.DataFrame([input_data])

    categorical_columns = ["crop", "zone", "condition"]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    expected_columns = model.feature_names_in_

    df = df.reindex(columns=expected_columns, fill_value=0)

    return df


def prepare_rainfall_input(input_data: dict, model) -> pd.DataFrame:
    missing_sensors = [
        sensor for sensor in RAINFALL_REQUIRED_SENSORS
        if sensor not in input_data
    ]

    if missing_sensors:
        raise ValueError(
            f"Faltan sensores para predecir rainfall: {missing_sensors}"
        )

    df = pd.DataFrame([input_data])

    categorical_columns = ["zone", "condition"]

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    expected_columns = model.feature_names_in_

    df = df.reindex(columns=expected_columns, fill_value=0)

    return df


# =========================
# MENSAJES PARA GRAFANA
# =========================

def build_temperature_message(crop: str, value: float) -> str:
    if crop == "rice":
        if value < 20:
            return "Temperatura esperada baja para arroz"
        if value > 35:
            return "Temperatura esperada alta para arroz"
        return "Temperatura esperada dentro del rango adecuado para arroz"

    if crop == "sugar_cane":
        if value < 18:
            return "Temperatura esperada baja para caña de azúcar"
        if value > 35:
            return "Temperatura esperada alta para caña de azúcar"
        return "Temperatura esperada dentro del rango adecuado para caña de azúcar"

    return "Temperatura esperada calculada por el modelo"


def build_rainfall_message(value: float) -> str:
    if value < 2:
        return "Precipitación esperada baja para arroz"
    if value > 70:
        return "Precipitación esperada alta para arroz"
    return "Precipitación esperada dentro del rango adecuado para arroz"


# =========================
# ESCRITURA A INFLUXDB
# =========================

def write_prediction_to_influx(
    parcel: str,
    crop: str,
    prediction_type: str,
    value: float,
    unit: str,
    message: str,
    model_version: str
):
    """
    Guarda la predicción en InfluxDB.
    Measurement: prediction_results
    """

    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )

    write_api = client.write_api(write_options=SYNCHRONOUS)

    point = (
        Point(PREDICTION_MEASUREMENT)
        .tag("parcel", parcel)
        .tag("crop", crop)
        .tag("prediction_type", prediction_type)
        .tag("model_name", "random_forest")
        .tag("model_version", model_version)
        .tag("unit", unit)
        .field("value", float(value))
        .field("message", message)
        .time(datetime.now(timezone.utc), WritePrecision.NS)
    )

    try:
        write_api.write(
            bucket=INFLUX_BUCKET,
            org=INFLUX_ORG,
            record=point
        )
    finally:
        client.close()


# =========================
# PREDICCIONES
# =========================

def predict_and_publish_temperature(parcel: str):
    model = joblib.load(TEMPERATURE_MODEL_PATH)

    latest_data = read_latest_sensor_values(parcel)

    crop = str(latest_data.get("crop", "unknown")).strip().lower()

    X_new = prepare_temperature_input(latest_data, model)

    prediction = float(model.predict(X_new)[0])

    message = build_temperature_message(crop, prediction)

    write_prediction_to_influx(
        parcel=parcel,
        crop=crop,
        prediction_type="temperature",
        value=prediction,
        unit="°C",
        message=message,
        model_version="rf_temperature_latest"
    )

    print(
        f"[OK] Temperatura publicada | Parcela: {parcel} | "
        f"Cultivo: {crop} | Valor: {prediction:.2f} °C"
    )


def predict_and_publish_rainfall(parcel: str):
    model = joblib.load(RAINFALL_MODEL_PATH)

    latest_data = read_latest_sensor_values(parcel, crop_filter="rice")

    crop = str(latest_data.get("crop", "rice")).strip().lower()

    X_new = prepare_rainfall_input(latest_data, model)

    prediction = float(model.predict(X_new)[0])

    message = build_rainfall_message(prediction)

    write_prediction_to_influx(
        parcel=parcel,
        crop=crop,
        prediction_type="rainfall",
        value=prediction,
        unit="mm",
        message=message,
        model_version="rf_rainfall_rice_latest"
    )

    print(
        f"[OK] Rainfall publicado | Parcela: {parcel} | "
        f"Cultivo: {crop} | Valor: {prediction:.2f} mm"
    )


def run_prediction_cycle():
    """
    Ejecuta un ciclo completo:
    - Predice temperatura para todas las parcelas.
    - Predice rainfall solo para parcelas de arroz.
    """

    parcels_df = get_available_parcels()

    if parcels_df.empty:
        print("No se encontraron parcelas activas en InfluxDB.")
        return

    print("\n" + "=" * 70)
    print("NUEVO CICLO DE PREDICCIÓN")
    print(f"Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for _, row in parcels_df.iterrows():
        parcel = row["parcel"]
        crop = row["crop"]

        try:
            predict_and_publish_temperature(parcel)
        except Exception as error:
            print(f"[ERROR] Temperatura | Parcela: {parcel} | {error}")

        if crop == "rice":
            try:
                predict_and_publish_rainfall(parcel)
            except Exception as error:
                print(f"[ERROR] Rainfall | Parcela: {parcel} | {error}")

    print("Ciclo finalizado.")


def main():
    validate_environment()

    print("Publicador de predicciones iniciado.")
    print(f"InfluxDB URL: {INFLUX_URL}")
    print(f"Bucket: {INFLUX_BUCKET}")
    print(f"Measurement de entrada: {SOURCE_MEASUREMENT}")
    print(f"Measurement de salida: {PREDICTION_MEASUREMENT}")
    print(f"Intervalo: {PUBLISH_INTERVAL_SECONDS} segundos")

    while True:
        try:
            run_prediction_cycle()
        except Exception as error:
            print(f"[ERROR GENERAL] {error}")

        print(f"\nEsperando {PUBLISH_INTERVAL_SECONDS} segundos para el siguiente ciclo...\n")
        time.sleep(PUBLISH_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()