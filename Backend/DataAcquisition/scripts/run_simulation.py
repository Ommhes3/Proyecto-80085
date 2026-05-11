from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backend.DataAcquisition.parcels.parcel_simulator import ParcelSimulator
from Backend.DataAcquisition.mqtt.mqtt_publisher import MqttPublisher


def run_simulation(interval_seconds: int = 1):
    parcel_simulator = ParcelSimulator()

    mqtt_publisher = MqttPublisher(
        broker="localhost",
        port=1883,
        base_topic="agricultura/sensores"
    )

    mqtt_publisher.connect()

    try:
        while True:
            readings = parcel_simulator.simulate_all_sensor_readings()

            mqtt_publisher.publish_many(readings)

            print("-" * 80)
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("Simulación detenida por el usuario.")

    finally:
        mqtt_publisher.disconnect()


if __name__ == "__main__":
    run_simulation(interval_seconds=1)