from pathlib import Path
import sys
import time

# Permite ejecutar este archivo directamente sin problemas de imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backend.DataAcquisition.parcels.parcel_simulator import ParcelSimulator
from Backend.DataAcquisition.mqtt.mqtt_publisher import MqttPublisher


def build_sensor_key(reading: dict) -> str:
    """
    Construye una clave única para controlar el tiempo de muestreo
    de cada sensor por parcela.

    Ejemplo:
    rice:parcel_1:temperature
    """

    crop = reading.get("crop", "unknown_crop")
    parcel = reading.get("parcel", "unknown_parcel")
    sensor = reading.get("sensor", "unknown_sensor")

    return f"{crop}:{parcel}:{sensor}"


def should_publish(reading: dict, last_publish_times: dict, current_time: float) -> bool:
    """
    Verifica si una lectura debe publicarse según su tiempo de muestreo.
    """

    sensor_key = build_sensor_key(reading)
    sampling_interval = reading.get("sampling_interval_seconds", 60)

    last_publish_time = last_publish_times.get(sensor_key)

    # Si nunca se ha publicado, se publica de una vez
    if last_publish_time is None:
        last_publish_times[sensor_key] = current_time
        return True

    elapsed_time = current_time - last_publish_time

    if elapsed_time >= sampling_interval:
        last_publish_times[sensor_key] = current_time
        return True

    return False


def run_simulation(loop_interval_seconds: int = 1):
    """
    Ejecuta la simulación IoT.

    Cada segundo revisa las lecturas generadas, pero solo publica
    aquellas cuyo tiempo de muestreo ya se cumplió.
    """

    parcel_simulator = ParcelSimulator()

    mqtt_publisher = MqttPublisher(
        broker="localhost",
        port=1883,
        base_topic="agricultura/sensores"
    )

    last_publish_times = {}

    mqtt_publisher.connect()

    try:
        while True:
            current_time = time.monotonic()

            readings = parcel_simulator.simulate_all_sensor_readings()

            readings_to_publish = []

            for reading in readings:
                if should_publish(reading, last_publish_times, current_time):
                    readings_to_publish.append(reading)

            if readings_to_publish:
                mqtt_publisher.publish_many(readings_to_publish)

                print(
                    f"Lecturas publicadas en este ciclo: {len(readings_to_publish)}"
                )
            else:
                print("Sin lecturas para publicar en este ciclo")

            print("-" * 80)

            time.sleep(loop_interval_seconds)

    except KeyboardInterrupt:
        print("Simulación detenida por el usuario.")

    finally:
        mqtt_publisher.disconnect()


if __name__ == "__main__":
    run_simulation(loop_interval_seconds=1)