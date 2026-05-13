import pandas as pd
import random
from pathlib import Path
import kagglehub


class StormSimulator:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"
        self.extreme_path = self.data_dir / "storm_extreme.csv"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.extreme_data = self.load_or_create_extreme_dataset()

    def load_or_create_extreme_dataset(self):
        """
        Carga el dataset storm si ya existe.
        Si no existe, lo genera una sola vez desde el dataset original.
        """

        if self.extreme_path.exists() and self.extreme_path.stat().st_size > 0:
            try:
                print(f"Dataset storm cargado desde: {self.extreme_path}")
                df = pd.read_csv(self.extreme_path)

                if not df.empty:
                    return df

                print("El dataset storm existe, pero está vacío. Se generará nuevamente.")

            except pd.errors.EmptyDataError:
                print("El dataset storm está dañado o vacío. Se generará nuevamente.")

        else:
            print("No existe dataset storm válido. Se generará nuevamente.")

        dataset_path = self.download_dataset()
        original_data = self.load_dataset(dataset_path)
        extreme_data = self.get_extreme_rows(original_data)

        extreme_data.to_csv(self.extreme_path, index=False)
        print(f"Dataset storm guardado en: {self.extreme_path}")

        return extreme_data

    def download_dataset(self):
        """
        Descarga el dataset desde KaggleHub y retorna la ruta del archivo CSV.
        """

        dataset_folder = Path(
            kagglehub.dataset_download("rajeev86/soil-climate-data")
        )

        csv_files = list(dataset_folder.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                "No se encontró ningún archivo CSV en el dataset descargado"
            )

        return csv_files[0]

    def load_dataset(self, dataset_path):
        """
        Carga el dataset climático original.
        """

        df = pd.read_csv(dataset_path)

        if df.empty:
            raise ValueError("El dataset original está vacío")

        return df

    def get_extreme_rows(self, df):
        """
        Filtra filas con condiciones climáticas atípicas o riesgosas.
        """

        extreme_rows = df[
            (df["Temperature"] < 18) |
            (df["Temperature"] > 35) |
            (df["Humidity"] < 50) |
            (df["Humidity"] > 85) |
            (df["Rainfall"] > 1500) |
            (df["Soil_pH"] < 5.5) |
            (df["Soil_pH"] > 7.5)
        ].copy()

        if extreme_rows.empty:
            print("No se encontraron datos extremos. Se usará el dataset completo.")
            return df.copy()

        print(f"Muestra storm encontrada: {len(extreme_rows)} registros")

        return extreme_rows

    def get_random_storm_row(self):
        """
        Retorna una fila aleatoria del dataset storm.
        """

        random_index = random.randint(0, len(self.extreme_data) - 1)
        return self.extreme_data.iloc[random_index]

    def generate_storm_reading(self, crop: str = "unknown", parcel: str = "parcel_1"):
        """
        Genera una lectura simulada usando condiciones atípicas.
        """

        row = self.get_random_storm_row()

        reading = {
            "parcel": parcel,
            "crop": crop,
            "simulator": "storm",
            "dataset_crop_reference": row.get("Crop_Type"),
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

    def export_extreme_dataset(self, output_path=None):
        """
        Exporta el dataset storm generado.
        """

        if output_path is None:
            output_path = self.extreme_path

        self.extreme_data.to_csv(output_path, index=False)
        print(f"Dataset storm exportado en: {output_path}")


if __name__ == "__main__":
    simulator = StormSimulator()

    reading = simulator.generate_storm_reading(
        crop="rice",
        parcel="parcel_1"
    )

    print(reading)