import pandas as pd
import random
from pathlib import Path
import kagglehub


class StormSimulator:
    def __init__(self):
        self.dataset_path = self.download_dataset()
        self.data = self.load_dataset()

    def download_dataset(self):
        """
        Descarga el dataset desde KaggleHub y retorna la ruta del archivo CSV.
        """

        dataset_folder = Path(
            kagglehub.dataset_download("rajeev86/soil-climate-data")
        )

        csv_files = list(dataset_folder.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("No se encontró ningún archivo CSV en el dataset descargado")

        return csv_files[0]

    def load_dataset(self):
        """
        Carga el dataset climático en un DataFrame.
        """

        df = pd.read_csv(self.dataset_path)

        if df.empty:
            raise ValueError("El dataset está vacío")

        return df

    def get_extreme_rows(self):
        """
        Filtra filas con condiciones climáticas atípicas o riesgosas.
        """

        df = self.data

        extreme_rows = df[
            (df["Temperature"] < 18) |
            (df["Temperature"] > 35) |
            (df["Humidity"] < 50) |
            (df["Humidity"] > 85) |
            (df["Rainfall"] > 1500) |
            (df["Soil_pH"] < 5.5) |
            (df["Soil_pH"] > 7.5)
        ]

        if extreme_rows.empty:
            return df

        return extreme_rows

    def get_random_storm_row(self):
        """
        Retorna una fila aleatoria con condiciones extremas.
        """

        extreme_rows = self.get_extreme_rows()
        random_index = random.randint(0, len(extreme_rows) - 1)

        return extreme_rows.iloc[random_index]

    def generate_storm_reading(self, crop: str = "unknown", parcel: str = "parcel_1"):
        """
        Genera una lectura simulada tipo sensor.
        """

        row = self.get_random_storm_row()

        reading = {
            "parcel": parcel,
            "crop": crop,
            "simulator": "storm",
            "crop_type_dataset": row.get("Crop_Type"),
            "soil_type": row.get("Soil_Type"),
            "temperature": round(float(row["Temperature"]), 2),
            "humidity": round(float(row["Humidity"]), 2),
            "rainfall": round(float(row["Rainfall"]), 2),
            "soil_ph": round(float(row["Soil_pH"]), 2),
            "irrigation_available": int(row["Irrigation_Available"]),
            "compatible": int(row["Compatible"]),
            "alert": True
        }

        return reading


if __name__ == "__main__":
    simulator = StormSimulator()

    reading = simulator.generate_storm_reading(
        crop="rice",
        parcel="parcel_1"
    )

    print(reading)