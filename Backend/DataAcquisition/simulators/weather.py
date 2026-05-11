import pandas as pd
import random
from pathlib import Path
import kagglehub
from sklearn.linear_model import LinearRegression


class WeatherSimulator:
    def __init__(self, target_size: int = 5000):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"
        self.synthetic_path = self.data_dir / "weather_synthetic.csv"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.target_size = target_size
        self.synthetic_data = self.load_or_create_synthetic_dataset()

    def load_or_create_synthetic_dataset(self):
        """
        Carga el dataset sintético si ya existe.
        Si no existe, lo genera una sola vez y lo guarda en la carpeta data.
        """

        if self.synthetic_path.exists():
            print(f"Dataset sintético cargado desde: {self.synthetic_path}")
            df = pd.read_csv(self.synthetic_path)

            if df.empty:
                raise ValueError("El dataset sintético existe, pero está vacío")

            return df

        print("No existe dataset sintético. Generando por primera vez...")

        dataset_path = self.download_dataset()
        original_data = self.load_dataset(dataset_path)
        normal_data = self.get_normal_rows(original_data)
        synthetic_data = self.generate_synthetic_weather_data(
            normal_data,
            target_size=self.target_size
        )

        synthetic_data.to_csv(self.synthetic_path, index=False)
        print(f"Dataset sintético guardado en: {self.synthetic_path}")

        return synthetic_data

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
        Carga el dataset original en un DataFrame.
        """

        df = pd.read_csv(dataset_path)

        if df.empty:
            raise ValueError("El dataset original está vacío")

        return df

    def get_normal_rows(self, df):
        """
        Filtra datos normales usando condiciones favorables para arroz y caña.

        Rangos usados:
        - Temperatura: 20 - 30 °C
        - Humedad: 60 - 90 %
        - Rainfall: 200 - 1500
        - pH suelo: 5.5 - 7.0
        """

        normal_rows = df[
            (df["Temperature"] >= 20) &
            (df["Temperature"] <= 30) &
            (df["Humidity"] >= 60) &
            (df["Humidity"] <= 90) &
            (df["Rainfall"] >= 200) &
            (df["Rainfall"] <= 1500) &
            (df["Soil_pH"] >= 5.5) &
            (df["Soil_pH"] <= 7.0)
        ].copy()

        if normal_rows.empty:
            raise ValueError(
                "No se encontraron datos normales usando la intersección de arroz y caña"
            )

        print(f"Muestra normal encontrada: {len(normal_rows)} registros")

        return normal_rows

    def generate_synthetic_weather_data(self, normal_data, target_size=5000):
        """
        Genera datos normales sintéticos a partir de la muestra filtrada.

        Usa regresión lineal para estimar Rainfall a partir de:
        - Temperature
        - Humidity
        - Soil_pH
        """

        df = normal_data.copy()

        x = df[["Temperature", "Humidity", "Soil_pH"]]
        y = df["Rainfall"]

        model = LinearRegression()
        model.fit(x, y)

        synthetic_rows = []

        for _ in range(target_size):
            base_row = df.sample(1).iloc[0].copy()

            temperature = base_row["Temperature"] + random.uniform(-1.5, 1.5)
            humidity = base_row["Humidity"] + random.uniform(-4, 4)
            soil_ph = base_row["Soil_pH"] + random.uniform(-0.25, 0.25)

            temperature = max(20, min(30, temperature))
            humidity = max(60, min(90, humidity))
            soil_ph = max(5.5, min(7.0, soil_ph))

            predicted_rainfall = model.predict(
                pd.DataFrame(
                    [[temperature, humidity, soil_ph]],
                    columns=["Temperature", "Humidity", "Soil_pH"]
                )
            )[0]

            rainfall = predicted_rainfall + random.uniform(-80, 80)
            rainfall = max(200, min(1500, rainfall))

            synthetic_row = base_row.copy()
            synthetic_row["Temperature"] = round(float(temperature), 2)
            synthetic_row["Humidity"] = round(float(humidity), 2)
            synthetic_row["Soil_pH"] = round(float(soil_ph), 2)
            synthetic_row["Rainfall"] = round(float(rainfall), 2)

            synthetic_rows.append(synthetic_row)

        return pd.DataFrame(synthetic_rows)

    def get_random_weather_row(self):
        """
        Retorna una fila aleatoria del dataset sintético.
        """

        random_index = random.randint(0, len(self.synthetic_data) - 1)
        return self.synthetic_data.iloc[random_index]

    def generate_weather_reading(self, crop: str = "unknown", parcel: str = "parcel_1"):
        """
        Genera una lectura simulada tipo sensor usando clima normal.
        """

        row = self.get_random_weather_row()

        reading = {
            "parcel": parcel,
            "crop": crop,
            "simulator": "weather",
            "dataset_crop_reference": row.get("Crop_Type"),
            "soil_type": row.get("Soil_Type"),
            "temperature": round(float(row["Temperature"]), 2),
            "humidity": round(float(row["Humidity"]), 2),
            "rainfall": round(float(row["Rainfall"]), 2),
            "soil_ph": round(float(row["Soil_pH"]), 2),
            "irrigation_available": int(row["Irrigation_Available"]),
            "compatible": int(row["Compatible"]),
            "alert": False
        }

        return reading

    def export_synthetic_dataset(self, output_path=None):
        """
        Exporta el dataset sintético generado.
        """

        if output_path is None:
            output_path = self.synthetic_path

        self.synthetic_data.to_csv(output_path, index=False)
        print(f"Dataset sintético exportado en: {output_path}")


if __name__ == "__main__":
    simulator = WeatherSimulator()

    reading = simulator.generate_weather_reading(
        crop="rice",
        parcel="parcel_1"
    )

    print(reading)