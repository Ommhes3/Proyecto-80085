from pathlib import Path
import sys
import os

import pandas as pd
import joblib

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG", "agro-iot")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "sensor_data")
MEASUREMENT = os.getenv("INFLUX_MEASUREMENT", "sensor_readings")


MODEL_PATH = (
    PROJECT_ROOT
    / "Backend"
    / "MachineLearning"
    / "outputs"
    / "random_forest_temperature_model_latest.joblib"
)


REQUIRED_SENSORS = [
    "humidity",
    "solar_radiation",
    "soil_ph",
]


def read_latest_sensor_values(parcel: str) -> dict:
    """
    Consulta InfluxDB y obtiene la última lectura disponible de cada sensor
    necesario para predecir temperatura en una parcela específica.
    """

    if not INFLUX_TOKEN:
        raise ValueError("No se encontró INFLUX_TOKEN. Revisa el archivo .env.")

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "{MEASUREMENT}")
      |> filter(fn: (r) => r._field == "value")
      |> filter(fn: (r) => r.parcel == "{parcel}")
      |> filter(fn: (r) => exists r.sensor and exists r.crop and exists r.zone and exists r.condition)
      |> group(columns: ["sensor"])
      |> last()
      |> keep(columns: ["_time", "_value", "sensor", "parcel", "crop", "zone", "condition"])
    '''

    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )

    result = client.query_api().query_data_frame(query)
    client.close()

    if isinstance(result, list):
        if not result:
            raise ValueError("InfluxDB no retornó datos para la parcela indicada.")
        df = pd.concat(result, ignore_index=True)
    else:
        df = result

    if df.empty:
        raise ValueError(f"No hay datos recientes para la parcela {parcel}.")

    latest_data = {}

    for _, row in df.iterrows():
        sensor = row["sensor"]
        value = row["_value"]
        latest_data[sensor] = value

    # Variables categóricas tomadas desde la última lectura consultada
    latest_data["crop"] = df.iloc[0]["crop"]
    latest_data["zone"] = df.iloc[0]["zone"]
    latest_data["condition"] = df.iloc[0]["condition"]

    return latest_data


def prepare_input_for_model(input_data: dict, model) -> pd.DataFrame:
    """
    Convierte la lectura tomada desde InfluxDB al formato esperado por el modelo.
    """

    missing_sensors = [
        sensor for sensor in REQUIRED_SENSORS
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


def predict_temperature_from_latest_data(parcel: str):
    """
    Carga el modelo entrenado y predice temperatura usando las últimas lecturas
    almacenadas en InfluxDB.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado: {MODEL_PATH}. "
            "Primero ejecuta random_forest_temperature.py"
        )

    model = joblib.load(MODEL_PATH)

    latest_data = read_latest_sensor_values(parcel)

    X_new = prepare_input_for_model(latest_data, model)

    prediction = model.predict(X_new)

    return prediction[0], latest_data


def main():
    parcel = input("Ingrese la parcela a predecir temperatura (ej. parcel_1): ").strip()

    if not parcel:
        print("Error: debe ingresar una parcela.")
        return

    if not parcel.startswith("parcel_"):
        print("Error: el formato esperado es parcel_x, por ejemplo parcel_1.")
        return

    try:
        predicted_temperature, latest_data = predict_temperature_from_latest_data(parcel)

        print("\nÚLTIMAS VARIABLES TOMADAS DESDE INFLUXDB")
        print("-" * 50)
        for key, value in latest_data.items():
            print(f"{key}: {value}")

        print("\nPREDICCIÓN DE TEMPERATURA")
        print("-" * 50)
        print(f"Parcela: {parcel}")
        print(f"Temperatura estimada: {predicted_temperature:.2f} °C")

    except Exception as error:
        print(f"Error al predecir temperatura para la parcela {parcel}: {error}")


if __name__ == "__main__":
    main()