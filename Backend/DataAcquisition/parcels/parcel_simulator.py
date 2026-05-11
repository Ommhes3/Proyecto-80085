import random
from datetime import datetime

from Backend.DataAcquisition.simulators.weather import WeatherSimulator
from Backend.DataAcquisition.simulators.storm import StormSimulator
from Backend.DataAcquisition.parcels.parcel_config import PARCELS


class ParcelSimulator:
    def __init__(self):
        self.weather_simulator = WeatherSimulator()
        self.storm_simulator = StormSimulator()

    def simulate_parcel_reading(self, parcel_config: dict) -> dict:
        """
        Simula una lectura para una parcela específica.

        Según la probabilidad configurada, la lectura puede venir del
        simulador de clima normal o del simulador de clima atípico.
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

    def simulate_all_parcels(self) -> list[dict]:
        """
        Simula una lectura para todas las parcelas configuradas.
        """

        readings = []

        for parcel_config in PARCELS:
            reading = self.simulate_parcel_reading(parcel_config)
            readings.append(reading)

        return readings


if __name__ == "__main__":
    simulator = ParcelSimulator()

    readings = simulator.simulate_all_parcels()

    for reading in readings:
        print(reading)