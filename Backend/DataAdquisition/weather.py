import pandas as pd
import random
from pathlib import Path
import kagglehub
from sklearn.linear_model import LinearRegression


class WeatherSimulator:
    def __init__(self):
        self.dataset_path = self.download_dataset()
        self.data = self.load_dataset()
        self.normal_data = self.get_normal_rows()
        self.synthetic_data = self.generate_synthetic_weather_data(target_size=5000)

    def download_dataset(self):
        dataset_folder = Path(
            kagglehub.dataset_download("rajeev86/soil-climate-data")
        )

        csv_files = list(dataset_folder.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError("No se encontró ningún archivo CSV en el dataset descargado")

        return csv_files[0]

    def load_dataset(self):
        df = pd.read_csv(self.dataset_path)

        if df.empty:
            raise ValueError("El dataset está vacío")

        return df

    def get_normal_rows(self):
        """
        Filtra los datos normales usando la intersección de condiciones favorables
        para arroz y caña de azúcar.

        Intersección:
        - Temperatura: 20 - 30 °C
        - Humedad: 60 - 90 %
        - pH suelo: 5.5 - 7.0
        - Rainfall: 200 - 1500 según escala del dataset
        """

        df = self.data

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

        return normal_rows

    def generate_synthetic_weather_data(self, target_size=5000):
        """
        Genera nuevos datos normales a partir de la muestra filtrada.

        Usa regresión lineal para estimar Rainfall a partir de:
        - Temperature
        - Humidity
        - Soil_pH

        Las demás columnas categóricas se copian desde registros reales.
        """

        df = self.normal_data.copy()

        print(f"Muestra normal encontrada: {len(df)} registros")

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

        synthetic_df = pd.DataFrame(synthetic_rows)

        return synthetic_df

    def get_random_weather_row(self):
        """
        Retorna una fila aleatoria del dataset sintético weather.
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
            "crop_type_dataset": row.get("Crop_Type"),
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

    def export_synthetic_dataset(self, output_path="weather_synthetic.csv"):
        """
        Exporta el dataset sintético generado.
        """

        self.synthetic_data.to_csv(output_path, index=False)
        print(f"Dataset sintético exportado en: {output_path}")


if __name__ == "__main__":
    simulator = WeatherSimulator()

    reading = simulator.generate_weather_reading(
        crop="rice",
        parcel="parcel_1"
    )

    print(reading)

    simulator.export_synthetic_dataset(
        "Backend/DataAdquisition/weather_synthetic.csv"
    )