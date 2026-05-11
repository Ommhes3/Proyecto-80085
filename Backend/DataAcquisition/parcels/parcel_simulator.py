import random
from datetime import datetime

from Backend.DataAcquisition.simulators.weather import WeatherSimulator
from Backend.DataAcquisition.simulators.storm import StormSimulator
from Backend.DataAcquisition.parcels.parcel_config import PARCELS
from Backend.DataAcquisition.parcels.sensor_config import SENSOR_CONFIG


class ParcelSimulator:
    def __init__(self):
        self.weather_simulator = WeatherSimulator()
        self.storm_simulator = StormSimulator()

    def generate_base_reading(self, parcel_config: dict) -> dict:
        """
        Genera una lectura base para una parcela.
        Primero decide si la parcela usa clima normal o clima atípico.
        """

        probability = random.random()

        if probability <= parcel_config["weather_probability"]:
            reading = self.weather_simulator.generate_weather_reading(
                crop=parcel_config["crop"],
                parcel=parcel_config["parcel"]
            )
            reading["condition"] = "normal"
        else:
            reading = self.storm_simulator.generate_storm_reading(
                crop=parcel_config["crop"],
                parcel=parcel_config["parcel"]
            )
            reading["condition"] = "storm"

        reading["timestamp"] = datetime.now().isoformat()
        reading["zone"] = parcel_config["zone"]

        return reading

    def simulate_solar_radiation(self, condition: str) -> float:
        """
        Simula radiación solar porque el dataset base no siempre trae esta variable.
        """

        if condition == "storm":
            return round(random.uniform(100, 260), 2)

        return round(random.uniform(300, 1000), 2)

    def simulate_wind_speed(self, condition: str) -> float:
        """
        Simula velocidad del viento, usada especialmente para caña de azúcar.
        """

        if condition == "storm":
            return round(random.uniform(20, 45), 2)

        return round(random.uniform(0, 20), 2)

    def normalize_rainfall(self, rainfall_value: float) -> float:
        """
        Ajusta la precipitación del dataset a una escala más manejable para el sensor.
        El dataset maneja valores altos, por eso se normaliza para simular mm/día.
        """

        return round(float(rainfall_value) / 20, 2)

    def get_sensor_value(self, sensor_name: str, base_reading: dict):
        """
        Extrae o simula el valor correspondiente a cada sensor.
        """

        if sensor_name == "temperature":
            return base_reading["temperature"]

        if sensor_name == "humidity":
            return base_reading["humidity"]

        if sensor_name == "rainfall":
            return self.normalize_rainfall(base_reading["rainfall"])

        if sensor_name == "soil_ph":
            return base_reading["soil_ph"]

        if sensor_name == "solar_radiation":
            return self.simulate_solar_radiation(base_reading["condition"])

        if sensor_name == "wind_speed":
            return self.simulate_wind_speed(base_reading["condition"])

        return None

    def is_sensor_alert(self, crop: str, sensor_name: str, value: float) -> bool:
        """
        Evalúa alertas según los umbrales definidos para cada cultivo.
        """

        if crop == "rice":
            if sensor_name == "temperature":
                return value < 20 or value > 35

            if sensor_name == "humidity":
                return value < 55 or value > 95

            if sensor_name == "rainfall":
                return value < 2 or value > 70

            if sensor_name == "solar_radiation":
                return value < 250

            if sensor_name == "soil_ph":
                return value < 5.5 or value > 7.5

        if crop == "sugar_cane":
            if sensor_name == "temperature":
                return value < 18 or value > 35

            if sensor_name == "humidity":
                return value < 50 or value > 92

            if sensor_name == "solar_radiation":
                return value < 250

            if sensor_name == "soil_ph":
                return value < 5.5 or value > 7.5

            if sensor_name == "wind_speed":
                return value > 20

        return False

    def build_sensor_readings(self, base_reading: dict) -> list[dict]:
        """
        Convierte una lectura base de parcela en varias lecturas por sensor.
        """

        crop = base_reading["crop"]
        sensor_config = SENSOR_CONFIG[crop]

        sensor_readings = []

        for sensor_name, config in sensor_config.items():
            value = self.get_sensor_value(sensor_name, base_reading)

            if value is None:
                continue

            sensor_reading = {
                "timestamp": base_reading["timestamp"],
                "parcel": base_reading["parcel"],
                "crop": base_reading["crop"],
                "zone": base_reading["zone"],
                "condition": base_reading["condition"],
                "simulator": base_reading["simulator"],

                "sensor": sensor_name,
                "sensor_type": config.get("sensor_type"),
                "value": value,
                "unit": config.get("unit"),
                "sampling_interval_seconds": config.get("sampling_interval_seconds"),

                "alert": self.is_sensor_alert(crop, sensor_name, value),

                "soil_type": base_reading.get("soil_type"),
                "dataset_crop_reference": base_reading.get("dataset_crop_reference")
            }

            sensor_readings.append(sensor_reading)

        return sensor_readings

    def simulate_parcel_sensor_readings(self, parcel_config: dict) -> list[dict]:
        """
        Simula todas las lecturas de sensores para una parcela.
        """

        base_reading = self.generate_base_reading(parcel_config)
        return self.build_sensor_readings(base_reading)

    def simulate_all_sensor_readings(self) -> list[dict]:
        """
        Simula las lecturas de sensores para todas las parcelas configuradas.
        """

        all_readings = []

        for parcel_config in PARCELS:
            parcel_readings = self.simulate_parcel_sensor_readings(parcel_config)
            all_readings.extend(parcel_readings)

        return all_readings

    def simulate_all_parcels(self) -> list[dict]:
        """
        Método de compatibilidad para no romper run_simulation.py.
        Ahora retorna lecturas individuales por sensor.
        """

        return self.simulate_all_sensor_readings()


if __name__ == "__main__":
    simulator = ParcelSimulator()

    readings = simulator.simulate_all_sensor_readings()

    for reading in readings:
        print(reading)