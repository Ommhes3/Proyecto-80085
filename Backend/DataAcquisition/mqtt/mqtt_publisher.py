import json
import paho.mqtt.client as mqtt


class MqttPublisher:
    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        base_topic: str = "agricultura/sensores"
    ):
        self.broker = broker
        self.port = port
        self.base_topic = base_topic
        self.client = mqtt.Client()

    def connect(self):
        """
        Conecta el cliente MQTT al broker.
        """
        self.client.connect(self.broker, self.port)
        self.client.loop_start()

        print(f"Conectado al broker MQTT en {self.broker}:{self.port}")

    def build_topic(self, reading: dict) -> str:
        """
        Construye el topic MQTT usando cultivo, parcela y sensor.

        Ejemplo:
        agricultura/sensores/rice/parcel_1/temperature
        """

        crop = reading.get("crop", "unknown_crop")
        parcel = reading.get("parcel", "unknown_parcel")
        sensor = reading.get("sensor", "unknown_sensor")

        return f"{self.base_topic}/{crop}/{parcel}/{sensor}"

    def publish_reading(self, reading: dict):
        """
        Publica una lectura individual de sensor en formato JSON.
        """

        topic = self.build_topic(reading)
        payload = json.dumps(reading, ensure_ascii=False)

        result = self.client.publish(topic, payload)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"Dato enviado a topic: {topic}")
            print(payload)
        else:
            print(f"Error publicando en topic: {topic}")

    def publish_many(self, readings: list[dict]):
        """
        Publica varias lecturas de sensores.
        """

        for reading in readings:
            self.publish_reading(reading)

    def disconnect(self):
        """
        Cierra la conexión MQTT.
        """
        self.client.loop_stop()
        self.client.disconnect()

        print("Conexión MQTT cerrada")